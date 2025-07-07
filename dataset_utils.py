import os
import pickle
import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils import load_config


np.random.seed(999)
random.seed(999)

configs = load_config('configs.json')

class SingleImageDataset(torch.utils.data.Dataset):
    
    '''
    Returns single images from the dataset (both pre- and post-images).
    '''
    
    def __init__(self, mode, configs, target_sen2_resolution=None, normalization_config=None):
        self.mode = mode
        self.configs = configs
        self.target_sen2_resolution = target_sen2_resolution
        self.normalization_config=normalization_config
        self.augmentation = configs['datasets']['augmentation']

        tmp = configs['dataset_type'].split('_')
        # format: "sen2_xx_mod_yy"
        source_types = [tmp[0], tmp[2]]
        sgd = {tmp[0]: tmp[1], tmp[2]: tmp[3]}

        self.source_types = source_types
        self.sgd = sgd

        candidate_paths = [i for i in Path(configs['paths']['dataset']).glob('*') if i.name == configs['dataset_type']]
        self.ds_path = candidate_paths[0]

        # Read the pickle files containing information on the splits
        patches = pickle.load(open(self.ds_path / configs['datasets'][mode], 'rb'))

        event_id = 0
        self.events = {}

        # Keep the positive indices in a separate list (useful for under/oversampling)
        self.positives_idx = []

        # Load the data paths into a dictionary
        for k in sorted(list(patches.keys())):
            # Load a MODIS and a Sentinel image both for pre and post
            try:
                if 'LR_image' in patches[k].keys():
                    self.events[event_id] = {'MODIS': patches[k]['LR_image'], 'SEN2': patches[k]['HR_image'], 'Event_ID': patches[k]['EVENT_ID']}
                    self.events[event_id]['key'] = k

                    if patches[k]['positive_flag']:
                        self.positives_idx.append(event_id)

                    event_id += 1
            except:
                print(k)
                exit(1)


        self.selected_bands = {}
        self.means = {}
        self.stds = {}
        for k, v in sgd.items():
            self.selected_bands[k] = configs['datasets']['selected_bands'][k].values()
            self.means[k] = [m for i, m in enumerate(configs['datasets'][f'{k}_mean'][v])]
            self.stds[k] = [m for i, m in enumerate(configs['datasets'][f'{k}_std'][v])]


    def downsample_modis(self, sample):
        '''
        Downsample the MODIS-related images to their original size.
        '''
        scaling_factor = int(self.sgd['mod']) // int(self.sgd['sen2'])
        resize_func = T.Resize(size=(self.configs['datasets']['img_size'] // scaling_factor, self.configs['datasets']['img_size'] // scaling_factor), interpolation=T.InterpolationMode.BICUBIC, antialias=True)

        interp_sample = sample.copy()
        interp_sample['MODIS_up'] = interp_sample['MODIS'].clone()

        if len(self.source_types) > 1:
            if interp_sample['MODIS'].ndim == 2:
                interp_sample['MODIS'] = resize_func(interp_sample['MODIS'][None, :, :])
            else:
                interp_sample['MODIS'] = resize_func(interp_sample['MODIS'])
        else:
            if sample.ndim == 2:
                interp_sample['img'] = resize_func(interp_sample['img'][None, :, :])
            else:
                interp_sample['img'] = resize_func(interp_sample['img'])

        return interp_sample

    def downsample_images(self, sample):

      sen2_scaling_factor = self.target_sen2_resolution / int(self.sgd['sen2'])
      sen2_resize_func = T.Resize(size=(int(self.configs['datasets']['img_size'] * sen2_scaling_factor),
                                        int(self.configs['datasets']['img_size'] * sen2_scaling_factor)),
                                  interpolation=T.InterpolationMode.BICUBIC, antialias=True)

      resize_func_up = T.Resize(size=(int(self.configs['datasets']['img_size']),int(self.configs['datasets']['img_size'])), interpolation=T.InterpolationMode.BICUBIC, antialias=True)


      sen2_resize_func = T.Resize(size=(int(self.configs['datasets']['img_size'] // sen2_scaling_factor),
                                        int(self.configs['datasets']['img_size'] // sen2_scaling_factor)),
                                  interpolation=T.InterpolationMode.BICUBIC, antialias=True)

      interp_sample = sample.copy()
      interp_sample['SEN2_down'] = interp_sample['SEN2'].clone()


      if interp_sample['SEN2'].ndim == 2:
          interp_sample['SEN2'] = sen2_resize_func(interp_sample['SEN2'][None, :, :])
      else:
          downsampled_img = sen2_resize_func(interp_sample['SEN2_down'])
          interp_sample['SEN2_down'] = resize_func_up(downsampled_img)

      return interp_sample



    def scale_img(self, sample_img, sample_name, scaling_mode):

        if scaling_mode == 'normalize':
            if 'SEN2' in sample_name:
                return TF.normalize(sample_img, mean=self.means['sen2'], std=self.stds['sen2'])
            elif 'MODIS' in sample_name:
                return TF.normalize(sample_img, mean=self.means['mod'], std=self.stds['mod'])

        elif scaling_mode == 'min-max':
            mins = sample_img.min(dim=-1).values.min(dim=-1).values
            maxs = sample_img.max(dim=-1).values.max(dim=-1).values
            print("mins:", mins)
            print("maxs:", maxs)

            uniq_mins = mins.unique()
            uniq_maxs = maxs.unique()
            if not (((len(uniq_mins) == 1) and (uniq_mins.item() == 0.)) and ((len(uniq_maxs) == 1) and (uniq_maxs.item() == 0.))):
                # Some images are all-zeros so scaling returns a NaN image
                new_ch = []
                for ch in range(sample_img.shape[0]):
                    if mins[ch] == maxs[ch]:
                        # Some channels contain only a single value, so scaling returns all-NaN
                        # We convert it to all-zeros
                        new_ch.append(torch.zeros(*sample_img[ch, :, :].shape)[None, :, :])
                    else:
                        new_ch.append(((sample_img[ch, :, :] - mins[:, None, None][ch]) / (maxs[:, None, None][ch] - mins[:, None, None][ch]))[None, :, :])

                return torch.cat(new_ch, dim=0)
        elif isinstance(scaling_mode, list):
            new_min, new_max = [torch.tensor(i) for i in scaling_mode]

            mins = sample_img.min(dim=-1).values.min(dim=-1).values
            maxs = sample_img.max(dim=-1).values.max(dim=-1).values

            uniq_mins = mins.unique()
            uniq_maxs = maxs.unique()
            if not (((len(uniq_mins) == 1) and (uniq_mins.item() == 0.)) and ((len(uniq_maxs) == 1) and (uniq_maxs.item() == 0.))):
                # Some images are all-zeros so scaling returns a NaN image
                new_ch = []
                for ch in range(sample_img.shape[0]):
                    if mins[ch] == maxs[ch]:
                        # Some channels contain only a single value, so scaling returns all-NaN
                        # We convert it to all-zeros
                        new_ch.append(torch.zeros(*sample_img[ch, :, :].shape)[None, :, :])
                    else:
                        new_ch.append(((sample_img[ch, :, :] - mins[:, None, None][ch]) / (maxs[:, None, None][ch] - mins[:, None, None][ch]))[None, :, :])

                return torch.mul(torch.cat(new_ch, dim=0), (new_max - new_min)) + new_min
        elif scaling_mode.startswith('clamp_scale'):
            thresh = int(scaling_mode.split('_')[-1])
            return torch.clamp(sample_img, min=0, max=thresh) / thresh
        elif scaling_mode.startswith('clamp'):
            thresh = int(scaling_mode.split('_')[-1])
            sample_img = torch.clamp(sample_img, min=0, max=thresh)

            if 'normalize' in scaling_mode:
                if 'SEN2' in sample_name:
                    return TF.normalize(sample_img, mean=self.means['sen2'], std=self.stds['sen2'])
                elif 'MODIS' in sample_name:
                    return TF.normalize(sample_img, mean=self.means['mod'], std=self.stds['mod'])

        # Squeeze back to original shape if necessary
        sample_img = sample_img.squeeze()

        return sample_img

    def scale_sample(self, sample):
        '''
        Scales the given images with the method defined in the config file.
        The input `sample` is a dictionary mapping image name -> image array.
        '''
        scaled_sample = sample.copy()

        for sample_name, sample_img in sample.items():
            if 'key' in sample_name:
                scaled_sample[sample_name] = sample_img
            elif 'SEN2' in sample_name:
                if self.configs['datasets']['scale_input_sen2'] is not None:
                    scaled_sample[sample_name] = self.scale_img(sample_img, sample_name, self.configs['datasets']['scale_input_sen2'])
                else:
                    scaled_sample[sample_name] = sample_img
            elif 'MODIS' in sample_name:
                if self.configs['datasets']['scale_input_mod'] is not None:
                    scaled_sample[sample_name] = self.scale_img(sample_img, sample_name, self.configs['datasets']['scale_input_mod'])
                else:
                    scaled_sample[sample_name] = sample_img

        return scaled_sample

    def final_normalization(self, sample):
        final_sample = sample.copy()

        if self.normalization_config is not None:
            for sample_name, sample_img in sample.items():
                if 'key' in sample_name: 
                    continue

                new_min = self.normalization_config['min']
                new_max = self.normalization_config['max']
                dtype = self.normalization_config['dtype']

                sample_min = sample_img.min()
                sample_max = sample_img.max()

                normalized_img = (sample_img - sample_min) / (sample_max - sample_min)
                scaled_img = normalized_img * (new_max - new_min) + new_min
                final_sample[sample_name] = scaled_img.to(dtype)

        return final_sample

    def load_img(self, sample):
        '''
        Loads an image.
        '''
        if len(self.source_types) > 1:
            loaded_sample = {}

            loaded_sample['MODIS'] = torch.load(sample['MODIS']).to(torch.float32)
            loaded_sample['SEN2'] = torch.load(sample['SEN2']).to(torch.float32)
        else:
            loaded_sample['img'] = torch.load(sample['img']).to(torch.float32)

        loaded_sample['key'] = f'{sample["key"]}'

        return loaded_sample


    def fillna(self, sample):
        '''
        Fills NaN values in the sample with the constant specified in the config.
        '''
        filled_sample = sample.copy()

        for sample_name, s in sample.items():
            if ('label' in sample_name) or ('key' in sample_name): continue
            filled_sample[sample_name] = torch.nan_to_num(s, nan=self.configs['datasets']['nan_value'])

        return filled_sample


    def augment(self, sample):
        '''
        Applies the following augmentations:
        - Random horizontal flipping (possibility = 0.5)
        - Random vertical flipping (possibility = 0.5)
        - Random Gaussian blurring (kernel size = 3) [only in train mode]
        '''
        aug_sample = sample.copy()

        # Horizontal flip
        if random.random() > 0.5:
            for sample_name, sample in aug_sample.items():
                if 'key' in sample_name: continue
                aug_sample[sample_name] = TF.hflip(sample)

        # Vertical flip
        if random.random() > 0.5:
            for sample_name, sample in aug_sample.items():
                if 'key' in sample_name: continue
                aug_sample[sample_name] = TF.vflip(sample)

        # Gaussian blur
        if (self.mode == 'train') and (random.random() > 0.5):
            for sample_name, sample in aug_sample.items():
                if ('label' in sample_name) or ('key' in sample_name): continue

                aug_sample[sample_name] = TF.gaussian_blur(sample, 3)

        return aug_sample


    def __len__(self):
        return len(self.events)


    def __getitem__(self, event_id):
        batch = self.events[event_id]

        # Load images
        batch = self.load_img(batch)

        # Replace NaN values with constant
        batch = self.fillna(batch)

        # Normalize images
        if (self.configs['datasets']['scale_input_sen2'] is not None) or (self.configs['datasets']['scale_input_mod'] is not None):
            batch = self.scale_sample(batch)

            # Some channels contain a single value (invalid) so scaling returns all NaN
            batch = self.fillna(batch)

        # Augment images
        if self.augmentation:
            batch = self.augment(batch)


        sen2_indices = list(self.selected_bands['sen2'])
        mod_indices = list(self.selected_bands['mod'])

        # Reordering the SEN2 tensor based on the indices
        batch['SEN2'] = batch['SEN2'][sen2_indices, :, :]

        # Reordering the MODIS tensor based on the indices
        batch['MODIS'] = batch['MODIS'][mod_indices, :, :]

        # Downsample MODIS images if needed
        if ('mod' in self.source_types) and self.configs['datasets']['original_modis_size']:
            batch = self.downsample_modis(batch)

        # Downsample images if needed
        if (self.target_sen2_resolution):
            batch = self.downsample_images(batch)

        if (self.normalization_config):
            batch = self.final_normalization(batch)  

        return batch
