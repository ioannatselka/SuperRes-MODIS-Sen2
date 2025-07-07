import os
import numpy as np
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json
import math
import sys
from tqdm import tqdm
import torchio as tio
import json
import matplotlib.pyplot as plt
from dataset_utils import SingleImageDataset
from models.srcnn import SRCNN
from models.fsrcnn import FSRCNN
from models.espcn import ESPCN
from models.vdsr import VDSR
from models.mrunet import MRUNet
from models.edsr import EDSR
from utils import normalize, correct_colors, load_best_model,load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configs = load_config('configs.json')

def plot_losses(path="checkpoints"):
    """
    Plot the training and validation losses stored in the checkpoint directory.

    Parameters:
    checkpoint_dir (str): Directory where the loss files are stored. Default is "checkpoints".

    Returns:
    None
    """
    #losses_path = os.path.join(checkpoint_dir, 'losses.json')
    losses_path = path

    # Read the losses from the JSON file
    with open(losses_path, 'r') as f:
        losses = json.load(f)

    train_losses = losses['train_losses']
    val_losses = losses['val_losses']

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
import json
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def plot_model_losses(model_name, losses_path):
    """
    Plot the training and validation losses stored in the checkpoint directory.

    Parameters:
    model_name (str): Name of the model.
    losses_path (str): Path to the JSON file containing the losses.

    Returns:
    None
    """
    # Read the losses from the JSON file
    with open(losses_path, 'r') as f:
        losses = json.load(f)

    train_losses = losses['train_losses']
    val_losses = losses['val_losses']

    return train_losses, val_losses

# Define model names and paths to their respective losses.json files
models_info = {
    'SRCNN': "/home/itselka/nvme1/itselka/code/results/srcnn/losses.json",
    'FSRCNN': "/home/itselka/nvme1/itselka/code/results/fsrcnn/losses.json",
    'EDSR': "/home/itselka/nvme1/itselka/results/edsr/losses.json",
    'ESPCN': "/home/itselka/nvme1/itselka/code/results/espcn/losses.json",
    'MRUNet': "/home/itselka/nvme1/itselka/code/results/mrunet/losses.json",
    'VDSR': "/home/itselka/nvme1/itselka/code/results/vdsr/losses.json"
}

# Set global font size
plt.rcParams.update({'font.size': 12})

# Create a figure and a 2x3 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(18, 20))

# Iterate through the models and their respective paths
for ax, (model_name, losses_path) in zip(axes.flat, models_info.items()):
    train_losses, val_losses = plot_model_losses(model_name, losses_path)
    
    # Plot the losses
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    ax.set_title(model_name, fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    ax.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Increase spacing between plots
plt.show()

selected_indices = [2, 3, 19, 40, 53, 56]  # Example list of selected indices

test_dataset = SingleImageDataset(mode='test', 
                             configs=configs, 
                             target_sen2_resolution=None, 
                             normalization_config={'min': 0, 'max': 1, 'dtype': torch.float32}, 
                             selected_indices=selected_indices
                             )

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)


# Define the loss function
mse_loss = nn.MSELoss()

def visualize_results(models, model_names, test_loader, device, num_samples=6):
    """
    Visualize the outputs of multiple models compared to the ground truth.

    Parameters:
    models (list of torch.nn.Module): List of trained models.
    model_names (list of str): List of model names corresponding to the models.
    test_loader (DataLoader): DataLoader for the test data.
    device (torch.device): The device to run the evaluation on.
    num_samples (int): Number of samples to visualize. Default is 5.

    Returns:
    None
    """
    for model in models:
        model.eval()
    
    samples_shown = 0
    num_models = len(models)
    fig, axes = plt.subplots(num_samples, num_models + 3, figsize=(4 * (num_models + 2), 5 * num_samples))
    
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.0, wspace=0.0)

    # Set the main titles above each column
    # Set the main titles above each column
    titles = ['Input', 'Bilinear\nInterpolation'] + model_names + ['Original\nHR Image']
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=20, fontweight='bold', fontname='Calibri')

    with torch.no_grad():
        for batch in test_loader:
            if samples_shown >= num_samples:
                break

            inputs = batch['MODIS']
            labels = batch['SEN2']
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = [model(inputs) for model in models]

            for i in range(inputs.size(0)):
                if samples_shown >= num_samples:
                    break
                input_image = inputs[i].cpu().numpy().transpose(1, 2, 0)
                output_images = [output[i].cpu().numpy().transpose(1, 2, 0) for output in outputs]
                label_image = labels[i].cpu().numpy().transpose(1, 2, 0)

                input_image = correct_colors(input_image)
                output_images = [correct_colors(output_image) for output_image in output_images]
                label_image = correct_colors(label_image)

                label_image = increase_brightness(label_image, 0.1)

                #label_image = cv2.convertScaleAbs(label_image, alpha=alpha, beta=beta)

                downsampled_image = cv2.resize(input_image, (256 // 8 , 256 // 8), interpolation=cv2.INTER_NEAREST) #INTER_CUBIC interpolation

                bilinear_image = cv2.resize(downsampled_image, (256,256), interpolation=cv2.INTER_LINEAR) #INTER_CUBIC interpolation


                #rescale = tio.RescaleIntensity(out_min_max=2)
                #label_image = rescale(label_image)

                axes[samples_shown, 0].imshow(input_image)
                axes[samples_shown, 0].axis('off')

                axes[samples_shown, 1].imshow(bilinear_image)
                axes[samples_shown, 1].axis('off')

                for j, output_image in enumerate(output_images):
                    axes[samples_shown, j + 2].imshow(output_image)
                    axes[samples_shown, j + 2].axis('off')

                axes[samples_shown, num_models + 2].imshow(label_image)
                axes[samples_shown, num_models + 2].axis('off')

                samples_shown += 1
    
    plt.subplots_adjust(wspace=0.1, hspace=0.01)
    plt.show()

def increase_brightness(image,value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 1 - value
    v[v > lim] = 1
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

# Define a dictionary mapping model names to their respective constructors
model_constructors = {
    'SRCNN': SRCNN,
    'FSRCNN': lambda: FSRCNN(5, 1),
    'EDSR': EDSR,
    'ESPCN': lambda: ESPCN(5, 1),
    'MRUNet': lambda: MRUNet(res_down=True, n_resblocks=1, bilinear=0),
    'VDSR': VDSR
}

# Paths to the checkpoint directories
model_paths = {
    'SRCNN': "/home/itselka/nvme1/itselka/code/results/srcnn",
    'FSRCNN': "/home/itselka/nvme1/itselka/code/results/fsrcnn",
    'EDSR': "/home/itselka/nvme1/itselka/resultsx8/edsr",
    'ESPCN': "/home/itselka/nvme1/itselka/code/results/espcn",
    'MRUNet': "/home/itselka/nvme1/itselka/code/results/mrunet",
    'VDSR': "/home/itselka/nvme1/itselka/code/results/vdsr"
}

# Instantiate and load models
models = []
model_names = []
for name, constructor in model_constructors.items():
    model = constructor()
    model = load_best_model(model, checkpoint_dir=model_paths[name])
    models.append(model.to(device))
    model_names.append(name)

# Assuming test_loader is defined elsewhere
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Visualize the results
visualize_results(models, model_names, test_loader, device, num_samples=6)

