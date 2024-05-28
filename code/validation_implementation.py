# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:57:34 2024

@author: DAM1
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
from AudioDetectionDataset import AudioDetectionData
from custom_collate import custom_collate
from validation import validation


val_d1 = DataLoader(AudioDetectionData(csv_file='../labeled_data/train_val_test_annotations/val.csv'),
                      batch_size=1,
                      shuffle = False,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)


# and then to load the model :    

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 6 # three classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
model.load_state_dict(torch.load('../models/WhaleMoanDetector_15.pth'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.eval()
model.to(device)

map_value = validation(val_d1, device, model)
print(f'Validation mAP: {map_value}')
