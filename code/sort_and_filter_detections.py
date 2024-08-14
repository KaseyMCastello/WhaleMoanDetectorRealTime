# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:29:06 2024

@author: Michaela Alksne
"""
#sort and filter detections from SOCAL34N and CC2008 
# sometimes there is accidental overlap in labels. get rid of that 

# read in modified annotations

import pandas as pd
import glob
import os

directory_path = "L:/WhaleMoanDetector/labeled_data/train_val_test_annotations"
files = glob.glob(os.path.join(directory_path, '*.csv'))


def flag_invalid_bboxes(df):
    flags = []
    for _, row in df.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        
        # Check if width and height are positive
        if xmax - xmin <= 0 or ymax - ymin <= 0:
            flags.append(True)
        else:
            flags.append(False)
            
    flagged_df = df.copy()
    flagged_df['invalid_bbox'] = flags
    return flagged_df

# Apply the remove_overlaps function to each group of 'audio_file' and 'annotation'
#cleaned_data = data.groupby(['audio_file', 'annotation']).apply(remove_overlaps).reset_index(drop=True)
total_invalid_count = 0


for file in files:
    # Read the CSV file
    data = pd.read_csv(file)
    
    # Validate and correct bounding boxes
    flagged_data = flag_invalid_bboxes(data)
    
    # Count invalid bounding boxes
    invalid_count = flagged_data['invalid_bbox'].sum()
    total_invalid_count += invalid_count
    
    # Print the count of invalid bounding boxes for this file
    print(f"File: {file}, Invalid bounding boxes: {invalid_count}")
    
    # Remove rows with invalid bounding boxes
    valid_data = flagged_data[~flagged_data['invalid_bbox']]
    
    # Drop the invalid_bbox column
    valid_data = valid_data.drop(columns=['invalid_bbox'])
    
    # Save the cleaned data back to CSV
    valid_data.to_csv(file, index=False)