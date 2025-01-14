# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:35:17 2024

@author: Michaela Alksne

plot bounding box annotations generated in Python for faster-rCNN 
runs in spyder
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display  # Import the display function
import os

base_dir = 'L:\\WhaleMoanDetector\\labeled_data\\train_val_test_annotations\\val.csv'

# Load annotations
#annotations_path = os.path.join(base_dir, 'labeled_data', 'spectrograms', 'HARP', 'SOCAL26H_annotations.csv')
annotations = pd.read_csv(base_dir)

# Function to plot bounding boxes and labels on the spectrograms
def plot_annotated_spectrograms(annotations,base_dir):
    grouped_annotations = annotations.groupby('spectrogram_path')
    
    for spectrogram_path, group in grouped_annotations:
        print(base_dir)
        print(spectrogram_path)
        full_spectrogram_path = os.path.join(base_dir,"WhaleMoanDetector",spectrogram_path)
        # Load the spectrogram image
        image = Image.open(full_spectrogram_path)
        draw = ImageDraw.Draw(image)  # Create a drawing context
        font = ImageFont.truetype("arial.ttf", 16)  # Adjust the font and size as needed

        # Plot each bounding box and label for this spectrogram
        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            label = row['label']
            # Draw rectangle on the image
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='black', width=3)
            # Place a label near the top-left corner of the bounding box, adjust positioning as needed
            #draw.text((xmin, ymin-17), label, fill='black', font = font)
        
        file_name = os.path.basename(spectrogram_path)

        # Display the image with bounding boxes and labels
        draw.text((0, 10), "File Path: " + file_name, fill='black', font = font)

        display(image)

# Call the function with the annotations dataframe
plot_annotated_spectrograms(annotations,base_dir)

annotations.label.value_counts()

