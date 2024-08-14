# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:33:42 2024

@author: Michaela Alksne

big long function for validation during training

prints aP and mAP and saves mAP. Then you can evaluate performance as a function of epoch. 
"""
import torchvision
from performance_metrics_functions import calculate_detection_metrics, calculate_precision_recall, calculate_ap

def validation(vald1, device, model):
    
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
    for data in vald1:
        
        img = data[0][0]
        detection_dict = data[1][0]  # Access the first dictionary in the list, which contains detection data

        # Now access the 'boxes' and 'labels' from the detection dictionary
        boxes = detection_dict['boxes']
        labels = detection_dict['labels']
        

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
              
                tp, fp, fn = calculate_detection_metrics(predictions_threshold, labels, category_id, iou_threshold, boxes)
                all_metrics[score_threshold][category_name]['tp'] += tp
                all_metrics[score_threshold][category_name]['fp'] += fp
                all_metrics[score_threshold][category_name]['fn'] += fn
    
    
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
        print(f"Validation Category: {category}, AP: {ap}")

    # Calculate mean AP (mAP)
    map_value = sum(ap_values) / len(ap_values)
    
    
    return map_value