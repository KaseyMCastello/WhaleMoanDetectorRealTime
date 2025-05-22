import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File path
path = r"L:\WhaleMoanDetector_predictions\CalCOFI_PR\CalCOFI_2013_04_verified_detections.txt"

# Load the dataset
data = pd.read_csv(path, sep="\t")

# Total calls per label
total_calls = data['label'].value_counts()

# Counts of TP, FP, FN per label
pr_counts = data.groupby(['label', 'pr']).size().unstack(fill_value=0)

# Combine the results into a single DataFrame
call_counts_summary = pr_counts.copy()
call_counts_summary['Total'] = total_calls

# Print the counts of calls, TP, FP, and FN
print("Counts of TP, FP, FN by Label:")
print(call_counts_summary)

# Define thresholds
thresholds = np.arange(0.00, 0.95, 0.05)

# Define the exact order of labels
ordered_labels = ['D', '40Hz', '20Hz', 'A', 'B']

# Initialize a dictionary to store precision-recall data
precision_recall = {}

# Calculate precision and recall for each ordered label
for label in ordered_labels:
    precision = []
    recall = []
    
    label_data = data[data['label'] == label]
    
    for threshold in thresholds:
        # Apply threshold
        thresholded_data = label_data[label_data['score'] >= threshold]
        
        # Calculate TP, FP, and FN
        tp = (thresholded_data['pr'] == 1).sum()
        fp = (thresholded_data['pr'] == 2).sum()
        fn = (label_data['pr'] == 3).sum()  # False negatives are independent of threshold
        
        # Calculate precision and recall
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision.append(precision_val)
        recall.append(recall_val)
    
    # Store the results
    precision_recall[label] = {"precision": precision, "recall": recall}

# Plot precision-recall curves
plt.figure(figsize=(10, 7))

for label in ordered_labels:
    pr_data = precision_recall[label]
    # Plot the curve with default colors
    plt.plot(pr_data["recall"], pr_data["precision"], label=f"Category {label}")
    
    # Add dots for each threshold
    plt.scatter(pr_data["recall"], pr_data["precision"], s=40)

# Add labels, legend, and axis limits with larger font sizes
#plt.title("Precision-Recall Curve for Each Category", fontsize=16)
plt.xlabel("Recall", fontsize=20)
plt.ylabel("Precision", fontsize=20)
plt.xticks(fontsize=16)  # Make x-axis tick labels larger
plt.yticks(fontsize=16)  # Make y-axis tick labels larger
# Move the legend to the bottom center
plt.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.25, 0.5), ncol=1)
plt.grid()
plt.xlim(0, 1)  # Ensure x-axis (recall) ranges from 0 to 1
plt.ylim(0, 1) # Ensure y-axis (precision) ranges from 0 to 1

# Save the plot to the specified directory
output_path = r"L:\WhaleMoanDetector\evaluation_preformance\CC2008-08-PR-curve-final-WMV.png"
plt.savefig(output_path, dpi=300)  # High-resolution save
plt.show()
plt.show()
