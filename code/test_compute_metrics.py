# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:01:09 2024

@author: Michaela Alksne

testing trained model on the test dataeset and computing IoU

for SONOBOI
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

test_d1 = DataLoader(AudioDetectionData_with_hard_negatives(csv_file = 'L:/WhaleMoanDetector/labeled_data/train_val_test_annotations/CC200808_test.csv'),
                      batch_size=1,
                      shuffle = True,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)



# and then to load the model :    

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 6 # three classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
model.load_state_dict(torch.load('../models/WhaleMoanDetector_8_26_24_4.pth'))
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

categories = {'D': D, '40Hz': fourtyHz, '20Hz': twentyHz, 'A': A, 'B': B}

score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
all_metrics = {thr: {cat: {'tp': 0, 'fp': 0, 'fn': 0} for cat in categories} for thr in score_thresholds}

tp_dict = {'D': 0, '40Hz': 0, '20Hz':0, 'A':0, 'B':0}
fp_dict = {'D': 0, '40Hz': 0, '20Hz':0, 'A':0, 'B':0}
fn_dict = {'D': 0, '40Hz': 0, '20Hz':0, 'A':0, 'B':0}

# Iterate over the test dataset
for data in test_d1:
    img = data[0][0].to(device)  # Move the image to the device
    # Check if ground truth boxes or labels are None (NaN images)
    if data[0][1] is None or data[0][1]["boxes"] is None or data[0][1]["labels"] is None:
        boxes = torch.empty((0, 4), device=device)  # Create an empty tensor for boxes
        labels = torch.empty((0,), dtype=torch.int64, device=device)  # Create an empty tensor for NaN spectrograms
    else:
        boxes = data[0][1]["boxes"].to(device)  # Move the boxes to the device
        labels = data[0][1]["labels"].to(device)  # Move the labels to the device
    
    # Run inference on the image
    output = model([img])
    
    # Get predicted bounding boxes, scores, and labels
    out_bbox = output[0]["boxes"]
    out_scores = output[0]["scores"]
    out_labels = output[0]["labels"]
    
    # Apply Non-Maximum Suppression
    keep = torchvision.ops.nms(out_bbox, out_scores, 0.1)
    out_bbox = out_bbox[keep]
    out_scores = out_scores[keep]
    out_labels = out_labels[keep]

    # Iterate over score thresholds
    for score_threshold in score_thresholds:
        # Filter predictions based on the score threshold
        threshold_mask = out_scores >= score_threshold
        predictions_threshold = {
            'boxes': out_bbox[threshold_mask],
            'labels': out_labels[threshold_mask],
            'scores': out_scores[threshold_mask]
        }

        # Loop over each category and calculate metrics
        for category_name, category_id in categories.items():
            # Ensure ground truth labels and boxes are on the same device as the predictions
            gt_boxes = boxes[labels == category_id].to(device)
            pred_boxes = predictions_threshold['boxes'][predictions_threshold['labels'] == category_id].to(device)

            tp, fp, fn = calculate_detection_metrics_JZ(predictions_threshold, labels, category_id, iou_threshold, boxes)
            all_metrics[score_threshold][category_name]['tp'] += tp
            all_metrics[score_threshold][category_name]['fp'] += fp
            all_metrics[score_threshold][category_name]['fn'] += fn

# Now `all_metrics` contains all the metrics for each category at each score threshold

# now plot precision and recall after computing all metrics

for category in categories.keys():
    precisions = []
    recalls = []
    
    for score_threshold in score_thresholds:
        metrics = all_metrics[score_threshold][category]
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        precision, recall = calculate_precision_recall(tp, fp, fn)
        precisions.append(precision)
        recalls.append(recall)
    
    plt.plot(recalls, precisions, marker='.', label=f'Category: {category}')
    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('../figures/WhaleMoanDetector_8_26_24_4_CalCOFI_2008_08.jpeg', format='jpeg')  # Save as SVG
plt.show()

# calculate AUC-PR (Area Under Precison-Recall Curve )
from sklearn.metrics import auc

auc_pr_dict = {}

for category in categories.keys():
    precisions = []
    recalls = []
    
    for score_threshold in score_thresholds:
        metrics = all_metrics[score_threshold][category]
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        precision, recall = calculate_precision_recall(tp, fp, fn)
        precisions.append(precision)
        recalls.append(recall)
    
    # Ensure that the recall values are sorted in ascending order with corresponding precision values
    recalls, precisions = zip(*sorted(zip(recalls, precisions)))
    
    # Calculate the AUC-PR
    auc_pr = auc(recalls, precisions)
    auc_pr_dict[category] = auc_pr
    print(f"Category: {category}, AUC-PR: {auc_pr}")


# calculate mAP

# Calculate AP for each category and store it
ap_values = []

for category in categories.keys():
    precisions = []
    recalls = []
    
    for score_threshold in score_thresholds:
        metrics = all_metrics[score_threshold][category]
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        precision, recall = calculate_precision_recall(tp, fp, fn)
        precisions.append(precision)
        recalls.append(recall)
    
    # Sort recalls and corresponding precisions
    recalls, precisions = zip(*sorted(zip(recalls, precisions)))
    
    # Calculate AP
    ap = calculate_ap(recalls, precisions)
    ap_values.append(ap)
    print(f"Category: {category}, AP: {ap}")

# Calculate mean AP (mAP)
map_value = sum(ap_values) / len(ap_values)
print(f"mAP: {map_value}")




