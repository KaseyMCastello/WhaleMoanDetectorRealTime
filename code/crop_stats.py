# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:45:11 2024

@author: Michaela Alksne

this is to extract all the crops and then compute some stats on them. if you cant see the call then remove the crop lol. 

this is now for the test data to compare true and false positive cropers and false neg!

compare the SNR vs score distribution of true pos vs all labels
compare the SNR vs score of false pos vs true pos
compare the SNR vs score distribution of false neg vs all labels

should probably have a "test" dataframe that has all of the predictions and whether they were labeled as true in the groudtruth

"""

import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor
from collections import defaultdict
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from AudioDetectionDataset import AudioDetectionData, AudioDetectionData_with_hard_negatives
import sklearn
from pprint import pprint
from IPython.display import display
from custom_collate import custom_collate
from performance_metrics_functions import calculate_detection_metrics_JZ,calculate_detection_metrics, calculate_precision_recall, calculate_ap

test_d1 = DataLoader(AudioDetectionData_with_hard_negatives(csv_file = '../labeled_data/train_val_test_annotations/test.csv'),
                      batch_size=1,
                      shuffle = True,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)



# and then to load the model :    

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 6 # three classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
model_path = 'L:/WhaleMoanDetector/models/WhaleMoanDetector_preprocessed_hard_negatives_epoch_3.pth'
model_name = os.path.basename(model_path)
model.load_state_dict(torch.load(model_path))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.eval()
model.to(device)

iou_values = []
iou_threshold = 0.1

D = 1 # aka D call
fourtyHz = 2 # 40 Hz call
twentyHz = 3
A = 4
B = 5


