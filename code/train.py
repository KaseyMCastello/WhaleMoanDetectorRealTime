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


train_d1 = DataLoader(AudioDetectionData_with_hard_negatives(csv_file='../labeled_data/train_val_test_annotations/train.csv'),
                      batch_size=16,
                      shuffle = True,
                      collate_fn = custom_collate, 
                      pin_memory = True if torch.cuda.is_available() else False)

                    
val_d1 = DataLoader(AudioDetectionData_with_hard_negatives(csv_file='../labeled_data/train_val_test_annotations/val.csv'),
                      batch_size=1,
                      shuffle = False,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)
        
        
model_name = "WhaleMoanDetector_04_04_25_"

# load our model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 6 # five classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)

# I might need to change in features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)  
        

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay= 0.0005) #SDG = stochastic gradient desent with these hyperparameters
num_epochs = 30

precision_recall_output = ""


# model training loop.
model.to(device)
for epochs in range(num_epochs):
    model.train() # Set the model to training mode
    epoch_train_loss  = 0
    for data in train_d1:
        imgs = []
        targets = []
        for d in data:
            imgs.append(d[0].to(device)) #we have to send each image from the dataloader to our cpu..
            #targ = {}
            
            if d[1] is None:  # Check if the target is None (hard negative example)
                # Create a dummy target for hard negatives with label 0
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
            
        loss_dict = model(imgs,targets)
        loss = sum(v for v in loss_dict.values())
        epoch_train_loss += loss.cpu().detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'training loss for epoch {epochs}:{epoch_train_loss}')
    
    # Build model name with epoch
    model_epoch_name = f"{model_name}_epoch{epochs}"
    
    model_save_path = f'../models/{model_epoch_name}.pth'
    torch.save(model.state_dict(), model_save_path)
    #validation
    
    model.eval()
    
    with torch.no_grad():
        precision_recall_output = validation(val_d1, device, model, epoch_train_loss, epochs, precision_recall_output)
        
        
        # Save the accumulated metrics
        precision_recall_file_path = f"L:/WhaleMoanDetector/evaluation/validation_{model_epoch_name}.txt"
        with open(precision_recall_file_path, "w") as f:
            f.write(precision_recall_output)

 
    
    
    
        
        
        
        
        
        
        
        
