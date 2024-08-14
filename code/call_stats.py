# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:37:23 2024

@author: First I should filter out totally unreasonable detections. 

First, make histograms of max and min frequency for each category
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the root folder containing the subfolders with CSV files
root_folder = 'L:/WhaleMoanDetector_predictions/epoch3'

# Predefined color mapping for call types 40Hz
color_mapping = {
    'D': '#1f77b4',
    '40Hz': '#ff7f0e',
    '20Hz': '#2ca02c',
    'A': '#d62728',
    'B': '#9467bd',
    # Add more call types and colors as needed
}

def process_csv_files(root_folder):
    all_detections = []

    # Traverse the folder structure
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                detections = pd.read_csv(file_path)
                all_detections.append(detections)
                
    # Concatenate all DataFrames
    combined_detections = pd.concat(all_detections, ignore_index=True)
    
    return combined_detections
    # Plot histograms of min and max frequencies for each category
    #plot_frequency_histograms(combined_detections)

def plot_frequency_histograms(detections):
    categories = detections['label'].unique()
    
    bin_edges = np.linspace(10, 150, 70)  # 20 bins from 10 to 150
    for category in categories:
        category_data = detections[detections['label'] == category]
        
        # Plot histogram for min frequency
        plt.figure(figsize=(10, 6))
        sns.histplot(category_data['min_frequency'], bins=bin_edges, color=color_mapping.get(category, '#000000'), label=category)       
        plt.xlabel('Min Frequency')
        plt.ylabel('Count')
        plt.title(f'Min Frequency Histogram for Category: {category}')
        plt.xlim(10, 150)
        plt.xticks(np.arange(10, 151, 10))  # Set ticks every 10 Hz
        plt.grid(True)
        plt.show() 
        
        # Plot histogram for max frequency
        plt.figure(figsize=(10, 6))
        sns.histplot(category_data['max_frequency'], bins=bin_edges, color=color_mapping.get(category, '#000000'), label=category)       
        plt.xlabel('Max Frequency')
        plt.ylabel('Count')
        plt.title(f'Max Frequency Histogram for Category: {category}')
        plt.xlim(10, 150)
        plt.xticks(np.arange(10, 151, 10))  # Set ticks every 10 Hz
        plt.grid(True)
        plt.show()

def max_frequency_vs_score(detections):
    categories = detections['label'].unique()

    for category in categories:
        category_data = detections[detections['label'] == category]
        
        # Plot max frequency vs score
        plt.figure(figsize=(10, 6))
        plt.scatter(category_data['max_frequency'], category_data['score'], color=color_mapping.get(category, '#000000'), label=category)
        plt.xlabel('Max Frequency')
        plt.ylabel('Score')
        plt.title(f'Max Frequency vs Score for Category: {category}')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.1))  # Assuming score ranges from 0 to 1
        plt.xticks(np.arange(10, 151, 10))  # Set ticks every 10 Hz
        plt.show()
    
    
# Define the root folder containing the subfolders with CSV file 

# Process all CSV files in the folder structure
combined_detections = process_csv_files(root_folder)

plot_frequency_histograms(combined_detections)
max_frequency_vs_score(combined_detections)

