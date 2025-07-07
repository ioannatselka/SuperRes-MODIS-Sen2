import os
import sys
import math
import random
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_config
import dataset_utils
from dataset_utils import SingleImageDataset
import models
from models.srcnn import SRCNN
from models.fsrcnn import FSRCNN
from models.espcn import ESPCN
from models.vdsr import VDSR
from models.mrunet import MRUNet
from models.edsr import EDSR


np.random.seed(999)
random.seed(999)


configs = load_config("configs.json")

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Get data sources and SGDs
tmp = configs['dataset_type'].split('_')
# format: "sen2_xx_mod_yy"
source_types = [tmp[0], tmp[2]]
sgd = {tmp[0]: tmp[1], tmp[2]: tmp[3]}

if configs['datasets']['select_bands'] == 'same':
    # Keep only the bands with the same wavelengths
    sen2_bands = {}
    mod_bands = {}
    for sen2_band in configs['datasets']['selected_bands']['sen2'].keys():
        sen2_band_idx = configs['datasets']['sen2_bands'][sgd['sen2']][sen2_band]
        if sen2_band in configs['datasets']['sen2_mod_500_band_mapping'].keys():
            sen2_bands[sen2_band] = sen2_band_idx
            mod_band = configs['datasets']['sen2_mod_500_band_mapping'][sen2_band]
            mod_bands[mod_band] = configs['datasets']['mod_bands'][sgd['mod']][mod_band]

    # Keep only the common SEN2-MODIS bands
    configs['datasets']['selected_bands']['sen2'] = sen2_bands
    configs['datasets']['selected_bands']['mod'] = mod_bands

print("Sentinel-2:", sen2_bands)
print("Modis:", mod_bands)


train_dataset = SingleImageDataset('train', configs)

val_dataset = SingleImageDataset('val', configs)


mse_loss = nn.MSELoss()


def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()
    print(end = ' ')
    if batch == batches:
        print()



def train_model(model, train_loader, val_loader, loss_func, optimizer, configs, device, checkpoint_dir="checkpoints"):
    """
    Train the given model with provided data loaders, loss function, and optimizer.

    Parameters:
    model (torch.nn.Module): The model to train.
    train_loader (DataLoader): DataLoader for the training data.
    val_loader (DataLoader): DataLoader for the validation data.
    loss_func (torch.nn.Module): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer.
    configs (dict): Dictionary containing training configurations.
    checkpoint_dir (str): Directory to save checkpoints and logs. Default is "checkpoints".

    Returns:
    None
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize the best validation loss to a large value
    best_val_loss = float('inf')

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []
    model.to(device)
    for epoch in range(configs['train']['n_epochs'] + 1):
        model.train()
        train_loss = 0.0
        for i,batch in enumerate(train_loader):
            inputs = batch['MODIS']
            labels = batch['SEN2']
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
      
        # Average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f"Epoch [{epoch}/{configs['train']['n_epochs']}], Training Loss: {train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['MODIS']
                labels = batch['SEN2']
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                
                val_loss += loss.item()
        
        # Average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        #print()
        print(f"Epoch [{epoch}/{configs['train']['n_epochs']}], Validation Loss: {val_loss}")
        print()

        # Check if the current validation loss is the best we have seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} with validation loss {val_loss}")

        # Save the losses to a file for plotting later
        losses_path = os.path.join(checkpoint_dir, 'losses.json')
        with open(losses_path, 'w') as f:
            json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

print("Start training for SRCNN")
model = SRCNN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="results/srcnn")
print("Finished training SRCNN")

  
print("Start training for ESPCN")
model = ESPCN(5,1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="results/espcn")
print("Finished training ESPCN")


print("Start training for FSRCNN")
model = FSRCNN(5,1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)


train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="results/fsrcnn")
print("Finished training FSRCNN")


print("Start training for VDSR")
model = VDSR()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="results/vdsr")
print("Finished training VDSR")


print("Start training for MRUNET")

model = MRUNet(res_down=True, n_resblocks=1, bilinear=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="results/mrunet")
print("Finished training MRUNET")


print("Start training for EDSR")
model = EDSR()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="results/edsr")
print("Finished training EDSR")

#      ---- Scaling Factor x4 ----

train_dataset = SingleImageDataset('train', configs, target_sen2_resolution = 120)

val_dataset = SingleImageDataset('val', configs, target_sen2_resolution =  120)


def train_model(model, train_loader, val_loader, loss_func, optimizer, configs, device, checkpoint_dir="checkpoints_x4"):
    """
    Train the given model with provided data loaders, loss function, and optimizer.

    Parameters:
    model (torch.nn.Module): The model to train.
    train_loader (DataLoader): DataLoader for the training data.
    val_loader (DataLoader): DataLoader for the validation data.
    loss_func (torch.nn.Module): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer.
    configs (dict): Dictionary containing training configurations.
    checkpoint_dir (str): Directory to save checkpoints and logs. Default is "checkpoints".

    Returns:
    None
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize the best validation loss to a large value
    best_val_loss = float('inf')

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []
    model.to(device)
    for epoch in range(configs['train']['n_epochs'] + 1):
        model.train()
        train_loss = 0.0
        for i,batch in enumerate(train_loader):
            inputs = batch['MODIS']
            labels = batch['SEN2_down']
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
       
        # Average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        #print()
        print(f"Epoch [{epoch}/{configs['train']['n_epochs']}], Training Loss: {train_loss}")
        #print()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['MODIS']
                labels = batch['SEN2_down']
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                
                val_loss += loss.item()
        
        # Average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        #print()
        print(f"Epoch [{epoch}/{configs['train']['n_epochs']}], Validation Loss: {val_loss}")
        print()

        # Check if the current validation loss is the best we have seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} with validation loss {val_loss}")

        # Save the losses to a file for plotting later
        losses_path = os.path.join(checkpoint_dir, 'losses.json')
        with open(losses_path, 'w') as f:
            json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)


print("Start training for SRCNN x4")
model = SRCNN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx4/srcnn")
print("Finished training SRCNN x4")

    
print("Start training for ESPCN x4")
model = ESPCN(5,1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx4/espcn")
print("Finished training ESPCN x4")


print("Start training for FSRCNN x4")
model = FSRCNN(5,1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx4/fsrcnn")
print("Finished training FSRCNN x4")



print("Start training for VDSR x4")
model = VDSR()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx4/vdsr")
print("Finished training VDSR x4")


print("Start training for MRUNET x4")
model = MRUNet(res_down=True, n_resblocks=1, bilinear=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx4/mrunet")
print("Finished training MRUNET x4")


print("Start training for EDSR x4")
model = EDSR()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx4/edsr")
print("Finished training EDSR x4")


#      ---- Scaling Factor x2 ----

train_dataset = SingleImageDataset('train', configs, target_sen2_resolution = 240)

val_dataset = SingleImageDataset('val', configs, target_sen2_resolution = 240)


def train_model(model, train_loader, val_loader, loss_func, optimizer, configs, device, checkpoint_dir="checkpointsx2"):
    """
    Train the given model with provided data loaders, loss function, and optimizer.

    Parameters:
    model (torch.nn.Module): The model to train.
    train_loader (DataLoader): DataLoader for the training data.
    val_loader (DataLoader): DataLoader for the validation data.
    loss_func (torch.nn.Module): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer.
    configs (dict): Dictionary containing training configurations.
    checkpoint_dir (str): Directory to save checkpoints and logs. Default is "checkpoints".

    Returns:
    None
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize the best validation loss to a large value
    best_val_loss = float('inf')

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []
    model.to(device)
    for epoch in range(configs['train']['n_epochs'] + 1):
        model.train()
        train_loss = 0.0
        for i,batch in enumerate(train_loader):
            inputs = batch['MODIS']
            labels = batch['SEN2_down']
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        
        # Average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        #print()
        print(f"Epoch [{epoch}/{configs['train']['n_epochs']}], Training Loss: {train_loss}")
        #print()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['MODIS']
                labels = batch['SEN2_down']
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                
                val_loss += loss.item()
        
        # Average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        #print()
        print(f"Epoch [{epoch}/{configs['train']['n_epochs']}], Validation Loss: {val_loss}")
        print()

        # Check if the current validation loss is the best we have seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} with validation loss {val_loss}")

        # Save the losses to a file for plotting later
        losses_path = os.path.join(checkpoint_dir, 'losses.json')
        with open(losses_path, 'w') as f:
            json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)



print("Start training for SRCNN x2")
model = SRCNN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx2/srcnn")
print("Finished training SRCNN x2")


    
print("Start training for ESPCN x2")
model = ESPCN(5,1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx2/espcn")
print("Finished training ESPCN x2")



print("Start training for FSRCNN x2")
model = FSRCNN(5,1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx2/fsrcnn")
print("Finished training FSRCNN x2")



print("Start training for VDSR x2")
model = VDSR()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx2/vdsr")
print("Finished training VDSR x2")



print("Start training for MRUNET")
model = MRUNet(res_down=True, n_resblocks=1, bilinear=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx2/mrunet")
print("Finished training MRUNET")



print("Start training for EDSR")
model = EDSR()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Need to set the images to be from 0-1 values and float
train_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
val_dataset.normalization_config = {'min': 0, 'max':1, 'dtype': torch.float32 }
train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

train_model(model, train_loader, val_loader, mse_loss, optimizer, configs, device, checkpoint_dir="resultsx2/edsr")
print("Finished training EDSR")