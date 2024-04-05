# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:01:09 2024

@author: Michaela Alksne

testing trained model on the test dataeset and computing IoU

 for SONOBOI
"""

import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
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
from torch import tensor
from collections import defaultdict
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys
sys.path.append(r"L:\Sonobuoy_faster-rCNN\code\PYTHON")
from AudioDetectionDataset_MNA_grayscale import AudioDetectionData
import sklearn
import torchmetrics
import pycocotools
from pprint import pprint
from IPython.display import display
from collections import OrderedDict


def custom_collate(data):# returns the data as is 
    return data 

csv_file_name = 'L:\\Sonobuoy_faster-rCNN\\labeled_data\\train_val_test_annotations\\SOCAL34N_CC2008_08_test.csv'

df = pd.read_csv(csv_file_name)

file_names = [os.path.basename(path) for path in df['spectrogram_path'].tolist()] # get a list of file names to go along with my plotting of them..
unique_filenames = list(OrderedDict.fromkeys(file_names))



test_d1 = DataLoader(AudioDetectionData(csv_file=csv_file_name),
                      batch_size=1,
                      shuffle = False,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)
        
# and then to load the model :    

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 6 # three classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)

# I might need to change in features.. tbd..
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
model.load_state_dict(torch.load('L:\\Sonobuoy_faster-rCNN\\trained_model\\Sonobuoy_model_epoch_14.pth'))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.eval()

font = ImageFont.truetype("arial.ttf", 16)  # Adjust the font and size as needed

# Iterate over the validation dataset in the dataloader
for i, data in enumerate(test_d1):
    img = data[0][0]
    boxes = data[0][1]["boxes"]
    labels = data[0][1]["labels"]
    
    filename = unique_filenames[i]

    
    # Run inference on the image
    output = model([img.to(device)])
    
    # Get predicted bounding boxes and scores
    outbbox = output[0]["boxes"]
    out_scores = output[0]["scores"]
    out_labels = output[0]["labels"]

    
    keep = torchvision.ops.nms(outbbox, out_scores, 0.2)  # Apply Non-Maximum Suppression based on IOU threshold. if something is less than this threshold, delete it 
    
    outbbox_keep = outbbox[keep]
    out_scores_keep = out_scores[keep]
    out_labels_keep = out_labels[keep]
    
    
    
    im = (img.permute(1, 2, 0).cpu().detach().numpy() * 255).astype('uint8')
   
    im = im.squeeze()
    
   
   # Create an Image object from the numpy array
    vsample = Image.fromarray(im, mode='L')
    # Create an ImageDraw object
    draw = ImageDraw.Draw(vsample)
    
    if isinstance(boxes, torch.Tensor) and boxes.numel() == 4:
       boxes = [boxes.tolist()]  # Convert single bounding box tensor to a list containing one bounding box
       for box, label in zip(boxes, labels):
           draw.rectangle(list(box)[0], fill=None, outline="white")
           #draw.text((box[0]-6, box[1]-15), f"Label: {label}", fill="white")

       for pred_box, pred_label, pred_score in zip(outbbox_keep, out_labels_keep, out_scores_keep):
           pred_score_formatted = round(pred_score.item(), 2)

          # for gt_box in boxes:
           #   iou = calculate_iou(pred_box.detach().cpu().numpy(), gt_box[0])
           #   iou_values.append(iou)  # Append IoU value to the list
           #   print("IoU:", iou)
           draw.rectangle(list(pred_box), fill=None, outline="black")
           draw.text((pred_box[0].item()-6, pred_box[1].item()-15), f"{pred_label},{pred_score_formatted}", fill="black", font=font)
           draw.text((10, 10), f"Filename: {filename}", fill="black", font=font)


          # for gt_box in boxes:
               
             #  iou = calculate_iou(pred_box, gt_box[0])
               #iou_torch = torchvision.ops.box_iou(pred_box, torch.tensor(gt_box[0]))
             #  print("IoU:", iou) 
               #print('IoU torch', iou_torch)
       
    else:
        for box, labels in zip(boxes, labels):
            draw.rectangle(list(box), fill=None, outline="white")
           # draw.text((box[0].item()-6, box[1].item()-15), f"Label: {label}", fill="white")
            
        for pred_box, pred_label, pred_score in zip(outbbox_keep, out_labels_keep, out_scores_keep):
            pred_score_formatted = round(pred_score.item(), 2)

          #  for gt_box in boxes:
          #     iou = calculate_iou(pred_box.detach().cpu().numpy(), gt_box)
           #    iou_values.append(iou)  # Append IoU value to the list
           #    print("IoU:", iou)
            draw.rectangle(list(pred_box), fill=None, outline="black")
            draw.text((pred_box[0].item()-6, pred_box[1].item() - 20), f"{pred_label},{pred_score_formatted}", fill="black", font=font)
            draw.text((10, 10), f"Filename: {filename}", fill="black", font=font)

  
  
    #vsample.show()
    
    display(vsample)
  
    


















