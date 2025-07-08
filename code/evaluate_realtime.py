import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for 2D time-frequency boxes.
    Each box = [start_time, end_time, low_freq, high_freq]
    """
    t1_start, t1_end, f1_low, f1_high = box1
    t2_start, t2_end, f2_low, f2_high = box2

    t_inter = max(0, (min(t1_end, t2_end) - max(t1_start, t2_start)).total_seconds())
    f_inter = max(0, (min(f1_high, f2_high) - max(f1_low, f2_low)))
    intersection = t_inter * f_inter

    area1 = (t1_end - t1_start).total_seconds() * (f1_high - f1_low)
    area2 = (t2_end - t2_start).total_seconds() * (f2_high - f2_low)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def match_predictions_to_gt(pred_df, gt_df, iou_threshold=0.5):
    """
    Match predictions to ground truth using IoU.
    Returns TP, FP, FN per label.
    """
    
    labels = sorted(set(gt_df['label']) | set(pred_df['label']))
    stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

    for label in labels:
        
        gt_label = gt_df[gt_df['label'] == label].copy()
        pred_label = pred_df[pred_df['label'] == label].copy()
        

        gt_label['matched'] = False
        pred_label['matched'] = False

        for pred_idx, pred_row in pred_label.iterrows():
            best_iou = 0
            best_gt_idx = None

            pred_box = [
                pd.Timestamp(pred_row['start_time']),
                pd.Timestamp(pred_row['end_time']),
                pred_row['low_f'],
                pred_row['high_f']
            ]

            
            for gt_idx, gt_row in gt_label.iterrows():
                if gt_label.at[gt_idx, 'matched']:
                    continue

                gt_box = [gt_row['abs_start_time'], gt_row['abs_end_time'], gt_row['low_f'], gt_row['high_f']]

                if(label == "d"):
                    print("Label d:")
                    print("Ground Truth: ", gt_box)
                    print("Prediction: ", pred_box)

                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                stats[label]['TP'] += 1
                gt_label.at[best_gt_idx, 'matched'] = True
                pred_label.at[pred_idx, 'matched'] = True
            else:
                stats[label]['FP'] += 1

        stats[label]['FN'] += (~gt_label['matched']).sum()

    return stats

def compute_metrics(stats):
    """
    Compute precision, recall, F1 per label and overall.
    """
    results = {}
    total_TP = total_FP = total_FN = 0

    for label, s in stats.items():
        TP, FP, FN = s['TP'], s['FP'], s['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results[label] = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'TP': TP, 'FP': FP, 'FN': FN
        }
        total_TP += TP
        total_FP += FP
        total_FN += FN

    # Overall metrics
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    results['Overall'] = {
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1': overall_f1,
        'TP': total_TP, 'FP': total_FP, 'FN': total_FN
    }

    return results


if __name__ == "__main__":
    # Load CSVs
    predictions = pd.read_csv(r"C:\Users\kasey\OneDrive - UC San Diego\Lab Work\RealTimeBaleen\Spectrograms_for_Compare\RPI_DecimetedTest\output.txt", sep='\t')
    ground_truth = pd.read_csv(r"C:\Users\kasey\OneDrive - UC San Diego\Lab Work\RealTimeBaleen\Ground_Truth_SOCAL34N_sitN_090722_200000.x.csv")

    # Optional renaming for clarity
    predictions.rename(columns={'box_y1': 'low_f', 'box_y2': 'high_f'}, inplace=True)

    # Reference datetime = 2009-07-22 20:00:00
    ref_time = datetime.strptime("2009-07-22 20:00:00", "%Y-%m-%d %H:%M:%S")

    # Compute absolute timestamps
    ground_truth['abs_start_time'] = ground_truth['start_time'].apply(lambda s: ref_time + timedelta(seconds=s))
    ground_truth['abs_end_time'] = ground_truth['end_time'].apply(lambda s: ref_time + timedelta(seconds=s))
    
    predictions['label'] = predictions['label'].str.strip().str.lower()
    ground_truth['label'] = ground_truth['label'].str.strip().str.lower()

    # Limit ground truth to events that start before end of predictions
    max_pred_time = predictions['end_time'].max()
    
    ground_truth = ground_truth[ground_truth['abs_start_time'] <= max_pred_time]
    
    # Compute statistics
    stats = match_predictions_to_gt(predictions, ground_truth, iou_threshold=0.5)
    metrics = compute_metrics(stats)

    # Print results
    for label, result in metrics.items():
        print(f"\nLabel: {label}")
        for k, v in result.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
