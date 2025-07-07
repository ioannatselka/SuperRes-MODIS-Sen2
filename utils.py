import os
import json
import torch
import numpy as np

"""
Normalization & Color Utilities
""" 
def normalize(band):
    band_min, band_max = band.min(), band.max()
    return (band - band_min) / (band_max - band_min)

def normalize_everything(image):
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        channel_min = channel.min()
        channel_max = channel.max()
        image[:, :, i] = (channel - channel_min) / (channel_max - channel_min)
    return image

def correct_colors(image):
    red_n = normalize(image[:, :, 0])
    green_n = normalize(image[:, :, 1])
    blue_n = normalize(image[:, :, 2])
    return np.stack([red_n, green_n, blue_n], axis=-1)

"""
Evaluation Helpers
"""
import numpy as np

def calculate_luminance(image):
    return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

def is_valid_number(x):
    return not np.isnan(x) and not np.isinf(x)

"""
Model I/O
"""
import os
import torch

def load_best_model(model, checkpoint_dir="checkpoints"):
    """
    Load the model with the lowest validation loss from the checkpoint directory.

    Parameters:
    model (torch.nn.Module): The model architecture to load the weights into.
    checkpoint_dir (str): Directory where the checkpoints are saved. Default is "checkpoints".

    Returns:
    torch.nn.Module: The model with the loaded weights.
    """
    # List all checkpoint files
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in directory {checkpoint_dir}")

    # Sort by epoch number, assuming filenames contain 'epoch_{number}.pth'
    checkpoints.sort(key=lambda x: int(x.split('_')[1][5:].split('.')[0]))  # Adjust this if your naming convention is different
    best_model_path = os.path.join(checkpoint_dir, checkpoints[-1])
    print(f"Loading best model from: {best_model_path}")

    # Determine the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model state dict with map_location to map storages to the selected device
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model

"""
Config Loader
"""

import json

def load_config(config_path="configs.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)