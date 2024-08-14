# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 07:39:50 2024

@author: MNA

Function to compute scaled snr metric on grayscale final images


"""

from PIL import Image
import numpy as np

def compute_snr(event):

    # Read in the spectrogram image
    image = Image.open(event['image_file_path']).convert('L')
    
    # Extract bounding box coordinates
    box_x1 = int(event['box_x1'])
    box_x2 = int(event['box_x2'])
    box_y1 = int(event['box_y1'])
    box_y2 = int(event['box_y2'])
    
    # Crop the image to the bounding box
    cropped_image = image.crop((box_x1, box_y2, box_x2, box_y1))
    
    # Convert the cropped image to a numpy array
    cropped_array = np.array(cropped_image)
    
    # Calculate the 75th percentile for the signal values (upper 25th %)
    signal_75th_percentile = np.percentile(cropped_array, 75)
    # Calculate the 25th percentile for the noise values (lower 25th %)
    noise_25th_percentile = np.percentile(cropped_image, 25)

    if noise_25th_percentile == 0:
        noise_25th_percentile = 1
    # Compute SNR
    snr = signal_75th_percentile / noise_25th_percentile

    return snr