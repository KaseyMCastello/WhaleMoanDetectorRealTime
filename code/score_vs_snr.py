# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 07:30:19 2024

@author: Michaela Alksne

score_vs_snr.py plots the relationship between score and snr in whatever dataset you run predictions on. 

We are starting with wav files and running inference pipeline and then calculating snr from the bounding boxes. 

We will at some point want to do this for the groundtruths too to compare their snrs and false positive snrs. but first i want to see if there is 
even a relationship (linear, sigmoid, poly, none) to be found between snr and score. and if certain epochs of the model have the relationship or dont?

"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Predefined color mapping for call types 40Hz
color_mapping = {
    'D': '#1f77b4',
    '40Hz': '#ff7f0e',
    '20Hz': '#2ca02c',
    'A': '#d62728',
    'B': '#9467bd',
    # Add more call types and colors as needed
}

def compute_snr(event):
    # Read in the spectrogram image
    image = Image.open(event['image_file_path']).convert('L')
    
    # Extract bounding box coordinates
    box_x1 = int(event['box_x1'])
    box_x2 = int(event['box_x2'])
    box_y1 = int(event['box_y1'])
    box_y2 = int(event['box_y2'])
   
    # Crop the image to the bounding box
    cropped_image = image.crop((box_x1, box_y1, box_x2, box_y2))
    
    # Convert the cropped image to a numpy array
    cropped_array = np.array(cropped_image)
  
    signal_max = np.max(cropped_array) + 1
    # Add one to each value to scale appropriately (KF advice)
    
    # Calculate the 25th percentile for the noise values (lower 25th %)
    noise_25th_percentile = np.percentile(cropped_array, 25) + 1

    # Compute SNR
    snr = signal_max / noise_25th_percentile

    return snr, cropped_image

def process_csv_files(root_folder):
    all_detections = []

    # Traverse the folder structure
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                detections = pd.read_csv(file_path)
                
                # Compute SNR for each detection and add it to the DataFrame
                detections['snr'] = detections.apply(lambda row: compute_snr(row)[0], axis=1)
                
                # Append the DataFrame to the list
                all_detections.append(detections)
                
    # Concatenate all DataFrames
    combined_detections = pd.concat(all_detections, ignore_index=True)
    
    return combined_detections

def plot_score_vs_snr(detections):
    categories = detections['label'].unique()
    for category in categories:
        category_data = detections[detections['label'] == category]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=category_data, x='snr', y='score', color=color_mapping.get(category, '#000000'), label=category) 
        plt.xlabel('SNR')
        plt.ylabel('Score')
        plt.title(f'Score vs SNR for Category: {category}')
        plt.grid(True)
        plt.show()

def plot_crops_with_snr(detections, score_threshold=.7, snr_threshold = 100):
    
    filtered_detections = detections[detections['score'] > score_threshold]
    filtered_detections = filtered_detections[filtered_detections['snr'] > snr_threshold]
    
    for _, event in filtered_detections.iterrows():
        
        snr, cropped_image = compute_snr(event)
        
        # Plot the cropped image with its SNR
        plt.figure(figsize=(6, 6))
        plt.imshow(cropped_image, cmap='gray', vmin=0, vmax=256)
        plt.title(f'Label: {event["label"]}, Score: {event["score"]}, SNR: {snr:.2f}')
        plt.axis('off')
        plt.show() 

# Define the root folder containing the subfolders with CSV files
root_folder = 'L:/WhaleMoanDetector_predictions/epoch3'

# Process all CSV files in the folder structure
combined_detections = process_csv_files(root_folder)

# Plot score vs SNR for each category
plot_score_vs_snr(combined_detections)

# Plot the cropped regions with SNR separately
plot_crops_with_snr(combined_detections)



