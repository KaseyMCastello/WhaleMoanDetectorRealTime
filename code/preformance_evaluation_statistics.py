# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 07:52:32 2024

@author: Michaela Alksne 

preformance evaluation statistics

computes precision and recall at score thresholds between 0.1 - 0.9

plots precision vs recall curve 

computes AP for each category and mAP

makes confusion matrix

"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from custom_collate import custom_collate
from AudioDetectionDataset import AudioDetectionData_with_hard_negatives

csv = 'L:/WhaleMoanDetector/labeled_data/train_val_test_annotations/CC200808_updated_test.csv'
# Define the subfolder where the figures will be saved
output_folder = "L:\\WhaleMoanDetector\\evaluation_preformance\\WMD_10_04_24\\epoch_10"

# Extract the base name without the file extension
base_name = os.path.splitext(os.path.basename(csv))[0]

# Load the test dataset
test_d1 = DataLoader(
    AudioDetectionData_with_hard_negatives(csv_file=csv),
    batch_size=1,
    shuffle=True,
    collate_fn=custom_collate,
    pin_memory=True if torch.cuda.is_available() else False
)


# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 6  # including background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('../models/WhaleMoanDetector_10_04_24_10.pth'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.eval().to(device)

# Constants
iou_threshold = 0.1
categories = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}
score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Initialize confusion matrix
confusion_mtx = np.zeros((len(categories), len(categories)), dtype=int)

# Map category indices for the confusion matrix
category_idx = {cat: idx for idx, cat in enumerate(categories.values())}

# Initialize metrics storage with integer keys
all_metrics = {
    thr: {cat: {'tp': 0, 'fp': 0, 'fn': 0} for cat in categories.values()} 
    for thr in score_thresholds
}

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
    output = model([img])[0]
    
    # Apply Non-Maximum Suppression
    keep = torchvision.ops.nms(output["boxes"], output["scores"], 0.1)
    out_bbox = output["boxes"][keep]
    out_scores = output["scores"][keep]
    out_labels = output["labels"][keep]
    
    # Loop over each score threshold
    for score_threshold in score_thresholds:
        valid_preds = out_scores > score_threshold
        filtered_boxes = out_bbox[valid_preds]
        filtered_labels = out_labels[valid_preds]
        
        #check for true positives. 
        if len(filtered_boxes) > 0 and len(boxes) > 0: 
            # If we have a true positive, our prediction (filtered boxes) will have something in it. AND our groundtruth (boxes) will have something in it.
            ious = box_iou(filtered_boxes, boxes) 
            # computes the IoU between every bounding box and groundtruth box in this image.
            
            # Loop through predictions to find matches with ground truth
            for i, pred_label in enumerate(filtered_labels): 
                # for this image, we loop through the predictions, where i is the index of the prediction we are looking at. 
                max_iou, max_iou_idx = ious[i].max(0) 
                # for the prediction of index i, this line finds the maximum IoU between it and all the groundtruths.
                gt_label = labels[max_iou_idx].item()
                
                # Update confusion matrix only if score_threshold is 0.5
                if score_threshold == 0.5:
                    if max_iou >= iou_threshold:
                        confusion_mtx[category_idx[gt_label], category_idx[pred_label.item()]] += 1  # Update confusion matrix
        
                
                if max_iou >= iou_threshold and labels[max_iou_idx] == pred_label:
                    all_metrics[score_threshold][pred_label.item()]['tp'] += 1
                else:
                    all_metrics[score_threshold][pred_label.item()]['fp'] += 1
            
            # Check for ground truth boxes not matched by predictions
            for j, gt_label in enumerate(labels):
                if ious[:, j].max(0)[0] < iou_threshold:
                    all_metrics[score_threshold][gt_label.item()]['fn'] += 1
        else:
            # If no predictions, all ground truth boxes are false negatives
            for gt_label in labels:
                all_metrics[score_threshold][gt_label.item()]['fn'] += 1

# Initialize lists to store precision and recall values for each category
pr_curves = {cat: {'precision': [], 'recall': []} for cat in categories.values()}

# Initialize a string to store precision and recall values for saving
precision_recall_output = ""
# Calculate and print precision and recall for each category and score threshold
for score_threshold in score_thresholds:
    print(f"Metrics for score threshold: {score_threshold}")
    precision_recall_output += f"Metrics for score threshold: {score_threshold}\n"  # Add score threshold to output
    for category_name, category_id in categories.items():
        tp = all_metrics[score_threshold][category_id]['tp']
        fp = all_metrics[score_threshold][category_id]['fp']
        fn = all_metrics[score_threshold][category_id]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_recall_output += f"Category {category_name}: Precision = {precision:.4f}, Recall = {recall:.4f}\n"
        print(f"Category {category_name}: Precision = {precision:.4f}, Recall = {recall:.4f}")
        
        # Append precision and recall values to pr_curves for this category
        pr_curves[category_id]['precision'].append(precision)
        pr_curves[category_id]['recall'].append(recall)

# Save precision and recall values to a text file
precision_recall_file_path = os.path.join(output_folder, f"{base_name}_precision_recall_output.txt")
with open(precision_recall_file_path, "w") as f:
    f.write(precision_recall_output)
    
# Plot the precision-recall curve for each category
plt.figure(figsize=(10, 7))
for category_name, category_id in categories.items():
    plt.plot(pr_curves[category_id]['recall'], pr_curves[category_id]['precision'], marker='.', label=f'Category {category_name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Category')
plt.legend(loc="lower left")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.savefig(os.path.join(output_folder, f"{base_name}_precision_recall_curve.png"), dpi=300)
plt.show()


# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", xticklabels=list(categories.keys()), yticklabels=list(categories.keys()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for 0.5 score threshold")
plt.savefig(os.path.join(output_folder, f"{base_name}_confusion_matrix_0_5.png"), dpi=300)
plt.show()

