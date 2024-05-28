# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:26:05 2024

@author: DAM1
"""

import torch
import torchvision

def calculate_detection_metrics(predictions, ground_truths, category, iou_threshold, boxes):
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
