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
import sys
sys.path.append(r"L:\Sonobuoy_faster-rCNN\code\PYTHON")
from AudioDetectionDataset import AudioDetectionData
import sklearn
import torchmetrics
import pycocotools
from pprint import pprint
from IPython.display import display

def custom_collate(data):# returns the data as is 
    return data 


val_d1 = DataLoader(AudioDetectionData(csv_file='L:\\Sonobuoy_faster-rCNN\\labeled_data\\train_val_test_annotations\\validation.csv'),
                      batch_size=1,
                      shuffle = True,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)

test_d1 = DataLoader(AudioDetectionData(csv_file='L:\\Sonobuoy_faster-rCNN\\labeled_data\\train_val_test_annotations\\test.csv'),
                      batch_size=1,
                      shuffle = True,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)
        
        
train_d1 = DataLoader(AudioDetectionData(csv_file='L:\\Sonobuoy_faster-rCNN\\labeled_data\\train_val_test_annotations\\train.csv'),
                      batch_size=1,
                      shuffle = True,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)


# and then to load the model :    

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 6 # three classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
model.load_state_dict(torch.load('L:\\Sonobuoy_faster-rCNN\\trained_model\\Sonobuoy_model_epoch_14.pth'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.eval()

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

def calculate_detection_metrics(predictions, ground_truths, category, iou_threshold):
    gt_indices = torch.where(ground_truths == category)
    gt_boxes = boxes[gt_indices]

    pred_boxes = predictions['boxes']
    pred_labels = predictions['labels']
    pred_scores = predictions['scores']

    num_gt = len(gt_indices[0])

    if pred_boxes.shape[0] == 0 or num_gt == 0:
        return (0, 0, num_gt)

    iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_boxes)

    true_pos = torch.sum(iou_matrix.max(1).values > iou_threshold).item()
    false_pos = pred_boxes.shape[0] - true_pos
    false_neg = num_gt - true_pos

    return (true_pos, false_pos, false_neg)


def calculate_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


def calculate_ap(recalls, precisions):
    """Calculate the Average Precision (AP) for a single category."""
    # Append sentinel values at the beginning and end
    recalls = [0] + list(recalls) + [1]
    precisions = [0] + list(precisions) + [0]
    
    # For each recall level, take the maximum precision found
    # to the right of that recall level. This ensures the precision
    # curve is monotonically decreasing.
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i-1] = max(precisions[i-1], precisions[i])
    
    # Calculate the differences in recall
    recall_diff = [recalls[i+1] - recalls[i] for i in range(len(recalls)-1)]
    
    # Calculate AP using the recall differences and precision
    ap = sum(precision * diff for precision, diff in zip(precisions[:-1], recall_diff))
    
    return ap



# Iterate over the test dataset
for data in test_d1:
    img = data[0][0]
    boxes = data[0][1]["boxes"]
    labels = data[0][1]["labels"]
    
    # Run inference on the image
    output = model([img.to(device)])
    
    # Get predicted bounding boxes, scores, and labels
    out_bbox = output[0]["boxes"]
    out_scores = output[0]["scores"]
    out_labels = output[0]["labels"]
    
    # Apply Non-Maximum Suppression
    keep = torchvision.ops.nms(out_bbox, out_scores, 0.1) # I think if we make this lower it will do better? idk. 
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
            tp, fp, fn = calculate_detection_metrics(predictions_threshold, labels, category_id, iou_threshold)
            all_metrics[score_threshold][category_name]['tp'] += tp
            all_metrics[score_threshold][category_name]['fp'] += fp
            all_metrics[score_threshold][category_name]['fn'] += fn

# Now `all_metrics` contains all the metrics for each category at each score threshold
print(all_metrics)


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
plt.savefig('L:/Sonobuoy_faster-rCNN/figures/precision_recall_curve.svg', format='svg')  # Save as SVG
plt.show()
plt.savefig('L:/Sonobuoy_faster-rCNN/figures/precision_recall_curve.svg', format='svg')  # Save as SVG


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




