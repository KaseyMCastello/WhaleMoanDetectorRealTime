# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:28:31 2024

@author: Michaela Alksne

plot predictions from inference. in spyder or command line. 

would be nice to add easy modification step. 

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display  # Import the display function
import os

base_dir = 'L:\WhaleMoanDetector_predictions\CalCOFI_2017\CalCOFI_2017_01\CalCOFI_2017_01_raw_detections.csv'

# Load annotations
#annotations_path = os.path.join(base_dir, 'labeled_data', 'spectrograms', 'HARP', 'SOCAL26H_annotations.csv')
predictions = pd.read_csv(base_dir)


# Call the function with the predictions dataframe

# Function to plot bounding boxes and labels on the spectrograms
def plot_annotated_spectrograms(predictions):
    
    # Group predictions by image file path
    grouped_predictions = predictions.groupby('image_file_path')
    
    # Loop through each spectrogram file
    for spectrogram_path, group in grouped_predictions:
        print(f"Base directory: {base_dir}")
        print(f"Spectrogram path: {spectrogram_path}")
        
        # Load the spectrogram image
        image = Image.open(spectrogram_path)
        draw = ImageDraw.Draw(image)  # Create a drawing context
        font = ImageFont.truetype("arial.ttf", 16)  # Adjust the font and size as needed

        # Plot each bounding box and label for this spectrogram
        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax = row['box_x1'], row['box_y1'], row['box_x2'], row['box_y2']
            label = row['label']
            score = row['score']
            
            # Draw rectangle on the image
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='black', width=3)
            
            # Format the label and score together
            label_text = f"{label} ({score:.2f})"
           
            # Optionally draw the label near the bounding box
            draw.text((xmin, ymin - 17), label_text, fill='black', font=font)
        
        file_name = os.path.basename(spectrogram_path)

        # Display the image with bounding boxes and labels
        draw.text((0, 10), "File Path: " + file_name, fill='black', font=font)
        image.show()
        
        # Pause to wait for user input to proceed to the next image
        input("Press Enter to continue to the next image...")

plot_annotated_spectrograms(predictions)
