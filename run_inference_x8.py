import os
import csv
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd 
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json
import math
import sys
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import piq
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
import numpy as np
from numpy import resize
from dataset_utils import SingleImageDataset
from models.srcnn import SRCNN
from models.fsrcnn import FSRCNN
from models.espcn import ESPCN
from models.vdsr import VDSR
from models.mrunet import MRUNet
from models.edsr import EDSR
from utils import (
    normalize, 
    correct_colors, 
    normalize_everything, 
    calculate_luminance, 
    is_valid_number, 
    load_best_model,
    load_config
)


np.random.seed(999)
random.seed(999)

configs = load_config('configs.json')

test_dataset = SingleImageDataset('test', configs)
test_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }

test_loader = DataLoader(test_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

mse_loss = nn.MSELoss()


def evaluate_model(model, test_loader, loss_func):
    """
    Evaluate the model on the test dataset and report the loss for each channel separately,
    as well as PSNR, SSIM, MS-SSIM, and IFC for the entire image and each channel.

    Parameters:
    model (torch.nn.Module): The trained model.
    test_loader (DataLoader): DataLoader for the test data.
    loss_func (torch.nn.Module): The loss function.

    Returns:
    dict: The average test loss, average test loss for each channel, PSNR, SSIM, MS-SSIM, and IFC.
    """
    model.eval()
    test_loss = 0.0
    channel_losses = None
    num_batches = len(test_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_psnr = 0.0
    total_ssim = 0.0
    total_ifc = 0.0

    channel_psnr = None
    channel_ssim = None
    channel_ifc = None

    luminance_loss = 0.0
    luminance_psnr = 0.0
    luminance_ssim = 0.0
    luminance_ifc = 0.0

    psnr_count = 0
    ssim_count = 0
    ifc_count = 0

    channel_psnr_count = None
    channel_ssim_count = None
    channel_ifc_count = None

    luminance_psnr_count = 0
    luminance_ssim_count = 0
    luminance_ifc_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = batch['MODIS']
            labels = batch['SEN2']
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Calculate overall loss
            loss = loss_func(outputs, labels)
            test_loss += loss.item()

            # Initialize channel-specific metrics
            if channel_losses is None:
                num_channels = labels.size(1)
                channel_losses = torch.zeros(num_channels)
                channel_psnr = torch.zeros(num_channels)
                channel_ssim = torch.zeros(num_channels)
                channel_ifc = torch.zeros(num_channels)
                
                channel_psnr_count = torch.zeros(num_channels)
                channel_ssim_count = torch.zeros(num_channels)
                channel_ifc_count = torch.zeros(num_channels)

            # Calculate loss and metrics per channel
            for i in range(num_channels):  # Iterate over channels
                channel_loss = loss_func(outputs[:, i, :, :], labels[:, i, :, :])
                channel_losses[i] += channel_loss.item()

                for j in range(outputs.size(0)):  # Iterate over batch size
                    output_img = outputs[j, i, :, :].cpu().numpy()
                    label_img = labels[j, i, :, :].cpu().numpy()

                    output_img = normalize(output_img)
                    label_img = normalize(label_img)

                    # PSNR for channel
                    psnr_value = skimage_psnr(label_img, output_img, data_range=1.0)
                    if is_valid_number(psnr_value):
                        channel_psnr[i] += psnr_value
                        channel_psnr_count[i] += 1

                    # SSIM for channel
                    ssim_value = skimage_ssim(label_img, output_img, data_range=1.0)
                    if is_valid_number(ssim_value):
                        channel_ssim[i] += ssim_value
                        channel_ssim_count[i] += 1


            # Calculate luminance loss and metrics
            for i in range(outputs.size(0)):  # Iterate over batch size
                output_img = outputs[i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
                label_img = labels[i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format

                output_luminance = calculate_luminance(output_img)
                label_luminance = calculate_luminance(label_img)

                luminance_loss_value = loss_func(torch.tensor(output_luminance).to(device), torch.tensor(label_luminance).to(device)).item()
                if is_valid_number(luminance_loss_value):
                    luminance_loss += luminance_loss_value

                # Normalize luminance images for SSIM and MS-SSIM
                norm_output_luminance = normalize(output_luminance)
                norm_label_luminance = normalize(label_luminance)

                # PSNR for luminance
                luminance_psnr_value = skimage_psnr(label_luminance, output_luminance, data_range=1.0)
                if is_valid_number(luminance_psnr_value):
                    luminance_psnr += luminance_psnr_value
                    luminance_psnr_count += 1

                # SSIM for luminance
                luminance_ssim_value = skimage_ssim(norm_label_luminance, norm_output_luminance, data_range=1.0)
                if is_valid_number(luminance_ssim_value):
                    luminance_ssim += luminance_ssim_value
                    luminance_ssim_count += 1


            # Calculate PSNR, SSIM, MS-SSIM, and IFC for the entire image
            for i in range(outputs.size(0)):  # Iterate over batch size
                output_img = outputs[i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
                label_img = labels[i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format

                # Normalize images for SSIM and MS-SSIM
                output_img = normalize_everything(output_img)
                label_img = normalize_everything(label_img)

                # PSNR
                total_psnr_value = skimage_psnr(label_img, output_img, data_range=1.0)
                if is_valid_number(total_psnr_value):
                    total_psnr += total_psnr_value
                    psnr_count += 1

                # SSIM
                total_ssim_value = skimage_ssim(label_img, output_img, multichannel=True, channel_axis=-1, data_range=1.0)
                if is_valid_number(total_ssim_value):
                    total_ssim += total_ssim_value
                    ssim_count += 1

        # Average the total loss, channel losses, and luminance loss
        test_loss /= num_batches
        channel_losses /= num_batches
        luminance_loss /= num_batches

        # Average PSNR, SSIM, MS-SSIM, and IFC
        avg_psnr = total_psnr / psnr_count
        avg_ssim = total_ssim / ssim_count

        # Average luminance-specific metrics
        avg_luminance_psnr = luminance_psnr / luminance_psnr_count
        avg_luminance_ssim = luminance_ssim / luminance_ssim_count

        # Average channel-specific metrics
        channel_psnr /= channel_psnr_count
        channel_ssim /= channel_ssim_count

        # Convert channel losses and metrics to lists for easier handling
        channel_losses = channel_losses.cpu().numpy().tolist()
        channel_psnr = channel_psnr.cpu().numpy().tolist()
        channel_ssim = channel_ssim.cpu().numpy().tolist()
        
        # Print overall metrics
        print("Overall Metrics:")
        print(f"Total Loss: {test_loss:.4f}")
        print(f"PSNR: {avg_psnr:.4f}")
        print(f"SSIM: {avg_ssim:.4f}")

        # Print luminance metrics
        print("\nLuminance Metrics:")
        print(f"  Loss: {luminance_loss:.4f}")
        print(f"  PSNR: {avg_luminance_psnr:.4f}")
        print(f"  SSIM: {avg_luminance_ssim:.4f}")

        # Print per-channel metrics
        print("\nPer-Channel Metrics:")
        for i in range(num_channels):
            print(f"Channel {i+1}:")
            print(f"  Loss: {channel_losses[i]:.4f}")
            print(f"  PSNR: {channel_psnr[i]:.4f}")
            print(f"  SSIM: {channel_ssim[i]:.4f}")

        return {
            'total_loss': test_loss,
            'channel_losses': channel_losses,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'luminance_loss': luminance_loss,
            'luminance_psnr': avg_luminance_psnr,
            'luminance_ssim': avg_luminance_ssim,
            'channel_psnr': channel_psnr,
            'channel_ssim': channel_ssim,
        }

def visualize_results(model, test_loader, device, num_samples=5):
    """
    Visualize the model's outputs compared to the ground truth.

    Parameters:
    model (torch.nn.Module): The trained model.
    test_loader (DataLoader): DataLoader for the test data.
    device (torch.device): The device to run the evaluation on.
    num_samples (int): Number of samples to visualize. Default is 5.

    Returns:
    None
    """
    model.eval()
    samples_shown = 0
    with torch.no_grad():
        for batch in test_loader:
            if samples_shown >= num_samples:
                break

            inputs = batch['MODIS']
            labels = batch['SEN2']
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            for i in range(inputs.size(0)):
                if samples_shown >= num_samples:
                    break
                input_image = inputs[i].cpu().numpy().transpose(1, 2, 0)
                output_image = outputs[i].cpu().numpy().transpose(1, 2, 0)
                print(output_image.shape)
                label_image = labels[i].cpu().numpy().transpose(1, 2, 0)

                # Normalize and correct colors for visualization
                input_image = correct_colors(input_image)
                output_image = correct_colors(output_image,)
                label_image = correct_colors(label_image,)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(input_image)
                axes[0].set_title('Input')
                axes[0].axis('off')

                axes[1].imshow(output_image)
                axes[1].set_title('Output')
                axes[1].axis('off')

                axes[2].imshow(label_image)
                axes[2].set_title('Ground Truth')
                axes[2].axis('off')

                plt.show()

                samples_shown += 1

model1 = FSRCNN(5,1)

model = load_best_model(model1, checkpoint_dir="/home/itselka/nvme1/itselka/results/fsrcnn")

model = model.to(device)
test_loss_dict = evaluate_model(model, test_loader, mse_loss)

model1 = SRCNN()

model = load_best_model(model1, checkpoint_dir="/home/itselka/nvme1/itselka/results/srcnn")

model = model.to(device)
test_loss_dict = evaluate_model(model, test_loader, mse_loss)

model1 = EDSR()

model = load_best_model(model1, checkpoint_dir="/home/itselka/nvme1/itselka/results/edsr")

model = model.to(device)
test_loss_dict = evaluate_model(model, test_loader, mse_loss)


model1 = ESPCN(5,1)

model = load_best_model(model1, checkpoint_dir="/home/itselka/nvme1/itselka/results/espcn")

model = model.to(device)
test_loss_dict = evaluate_model(model, test_loader, mse_loss)


model1 = MRUNet(res_down=True, n_resblocks=1, bilinear=0)

model = load_best_model(model1, checkpoint_dir="/home/itselka/nvme1/itselka/results/mrunet")

model = model.to(device)
test_loss_dict = evaluate_model(model, test_loader, mse_loss)

model1 = VDSR()

model = load_best_model(model1, checkpoint_dir="/home/itselka/nvme1/itselka/results/vdsr")

model = model.to(device)
test_loss_dict = evaluate_model(model, test_loader, mse_loss)