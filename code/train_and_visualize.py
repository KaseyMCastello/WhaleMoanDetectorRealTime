# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:01:09 2024

@author: Michaela Alksne

# script to train faster rCNN model 
# this is for the Sonobuoys!!!!

# the pretrained Faster R-CNN ResNet-50 model that we are going to use expects the input image tensor to be in the form [c, h, w], where:

# c is the number of channels, for RGB images its 3 (which is what I have rn)
# h is the height of the image
# w is the width of the image

"""

import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from AudioDetectionDataset import AudioDetectionData, AudioDetectionData_with_hard_negatives
from custom_collate import custom_collate
from validation import validation
from torchvision.ops import box_iou
import os
from functools import partial  # Import partial to fix the argument passing


# Hook function to extract feature maps from each layer and label the layers
# Hook function to extract feature maps and label the layers
# Hook function to extract feature maps and label the layers
def hook_fn(layer_name, feature_map_storage, module, input, output):
    """Stores feature map for the layer."""
    feature_map_storage[layer_name] = output
    print(f"{layer_name} - Feature map dimensions: {output.shape}")


train_d1 = DataLoader(AudioDetectionData_with_hard_negatives(csv_file='../labeled_data/train_val_test_annotations/train.csv'),
                      batch_size=16,
                      shuffle = False,
                      collate_fn = custom_collate, 
                      pin_memory = True if torch.cuda.is_available() else False)

                    
# val_d1 = DataLoader(AudioDetectionData_with_hard_negatives(csv_file='../labeled_data/train_val_test_annotations/val.csv'),
#                       batch_size=1,
#                       shuffle = False,
#                       collate_fn = custom_collate,
#                       pin_memory = True if torch.cuda.is_available() else False)
        
        
# load our model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 6 # five classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)

# I might need to change in features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)  

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
# Move model to the correct device (GPU or CPU)
model.to(device)
# Register hooks to capture feature maps from multiple layers
#layer_name_list = []  # List to store layer names
#model.backbone.body.conv1.register_forward_hook(hook_fn('conv1'))  # First layer
#model.backbone.body.layer1[0].register_forward_hook(hook_fn('ResNet Layer1 Block0'))  # First residual block
#model.backbone.body.layer2[0].register_forward_hook(hook_fn('ResNet Layer2 Block0'))  # Second residual block
#model.backbone.body.layer3[0].register_forward_hook(hook_fn('ResNet Layer3 Block0'))  # Third residual block
#model.backbone.body.layer4[0].register_forward_hook(hook_fn('ResNet Layer4 Block0'))  # Fourth residual block

# Visualizing the spectrograms after downsampling in the training loop
# Visualizing the spectrograms after downsampling in the training loop
# Visualizing the spectrograms after downsampling in the training loop
# Visualizing the spectrograms after downsampling in the training loop


# Visualizing the spectrograms after downsampling in the training loop
def visualize_feature_map(output, layer_name, epoch, batch_idx):
    """Visualize the first 64 filters of the layer, each with its original resolution."""
    # Normalize output feature map to the range [0, 1] for better visualization
    min_val = output.min()
    max_val = output.max()
    output = (output - min_val) / (max_val - min_val)  # Normalize

    num_features = output.shape[1]  # Number of filters (feature maps)
    print(f"Layer {layer_name}: {num_features} filters")

    # Display only the first 64 filters
    num_filters_to_display = min(64, num_features)  # Only display first 64 filters

    # Create a figure to display the filters
    fig = plt.figure(figsize=(20, 20))  # Dynamically adjust the figure size

    for i in range(num_filters_to_display):
        ax = fig.add_subplot(8, 8, i + 1)  # Create a subplot for each filter

        # Get the height and width of the feature map
        height, width = output.shape[2], output.shape[3]

        # Plot each feature map with its original resolution
        ax.imshow(output[0, i].cpu().detach().numpy(), cmap='gray', aspect='auto')

        # Set the title for each filter (optional)
        ax.set_title(f"Filter {i + 1}", fontsize=10)
        ax.axis('off')

    # Adjust the layout and add a title with the layer name
    plt.tight_layout()
    plt.suptitle(f"Epoch {epoch}, Batch {batch_idx}: {layer_name}", fontsize=16)
    plt.subplots_adjust(top=0.95)  # Adjust the title position to avoid overlap
    plt.show()

    
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay= 0.0005) #SDG = stochastic gradient desent with these hyperparameters
num_epochs = 30

precision_recall_output = ""


# Training loop with visualization
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    feature_map_storage = {}  # Dictionary to store feature maps for each layer
    for batch_idx, data in enumerate(train_d1):
        imgs = []
        targets = []
        for d in data:
            imgs.append(d[0].to(device))
            if d[1] is None:
                targ = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32).to(device),
                    'labels': torch.tensor([0], dtype=torch.int64).to(device)
                }
            else:
                targ = {
                    'boxes': d[1]['boxes'].to(device),
                    'labels': d[1]['labels'].to(device)
                }
            targets.append(targ)
            
        # Register hooks for each layer one by one, using partial to bind feature_map_storage
        model.backbone.body.conv1.register_forward_hook(partial(hook_fn, 'conv1', feature_map_storage))  # Hook for conv1
        model.backbone.body.layer1[0].register_forward_hook(partial(hook_fn, 'ResNet Layer1 Block0', feature_map_storage))  # Hook for Layer1 Block0
        model.backbone.body.layer2[0].register_forward_hook(partial(hook_fn, 'ResNet Layer2 Block0', feature_map_storage))  # Hook for Layer2 Block0
        model.backbone.body.layer3[0].register_forward_hook(partial(hook_fn, 'ResNet Layer3 Block0', feature_map_storage))  # Hook for Layer3 Block0
        model.backbone.body.layer4[0].register_forward_hook(partial(hook_fn, 'ResNet Layer4 Block0', feature_map_storage))  # Hook for Layer4 Block0


        loss_dict = model(imgs, targets)
        loss = sum(v for v in loss_dict.values())
        epoch_train_loss += loss.cpu().detach().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Visualize feature maps for each layer (one by one)
        for layer_name, feature_map in feature_map_storage.items():
            visualize_feature_map(feature_map, layer_name, epoch, batch_idx)
            
    print(f'training loss: {epoch_train_loss}')

    # Save the model after each epoch
    model_save_path = f'../models/WhaleMoanDetector_01_23_24_{epoch}.pth'
    torch.save(model.state_dict(), model_save_path)
    #validation
   
    
    
        
        
        
        
        
        
        
        
