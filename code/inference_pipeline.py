# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:41:34 2024

@author: Michaela Alksne

script to run the required functions in the correct order to make predictions using trained model
"""

import librosa
import numpy as np
import torch
import os
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.colors import Normalize
import torch
import pandas as pd
import torchvision.ops as ops
from PIL import ImageOps
from datetime import datetime, timedelta
from IPython.display import display
import csv
import yaml
from inference_functions import extract_wav_start, chunk_audio, audio_to_spectrogram, predict_and_save_spectrograms

# Load the config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
# Access the configuration variables
wav_directory = config['wav_directory']
csv_file_path = config['csv_file_path']
model_path = config['model_path']

# Define spectrogram and data parameters
fieldnames = ['wav_file_path', 'model_no', 'image_file_path', 'label', 'score',
              'start_time_sec','end_time_sec','start_time','end_time',
              'min_frequency', 'max_frequency','box_x1', 'box_x2', 
              'box_y1', 'box_y2' ]
model_name = os.path.basename(model_path)
visualize_tf = False
label_mapping = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
window_size = 60
overlap_size = 0
time_per_pixel = 0.1  # Since hop_length = sr / 10, this simplifies to 1/10 second per pixel

# Load trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 6 # 5 classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# Open the CSV file just once, and write headers
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
   
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
   
    writer.writeheader()

# Loop over each file in the directory or subdirectory
    for dirpath, dirnames, filenames in os.walk(wav_directory):
       
        for filename in filenames:
          
            if filename.endswith('.wav'):
            # Full path to the WAV file
                wav_file_path = os.path.join(dirpath, filename)
                
                # Extract the subdirectory name if exists
                subfolder = os.path.relpath(dirpath, wav_directory)
                
                if subfolder == '.':
                    # if subfolder exists, use this in audiobasepath name ( for saving images)
                    audio_basename = os.path.splitext(filename)[0]
                 
                    print(audio_basename)
                else:
                    # if no subfolder exists, use wav file name
                    audio_basename = os.path.splitext(os.path.basename(wav_file_path))[0]
                   
                    print(audio_basename)
                # Extract the start datetime from the WAV file
                wav_start_time = extract_wav_start(wav_file_path)  # Ensure this returns a datetime object
                #wav_start_time = extract_wav_start(path)
                is_xwav = filename.endswith('.x.wav') #check if it is an xwav or a wav file 
                # Process each WAV file as you have in your folder
                chunks, sr, chunk_start_times = chunk_audio(wav_file_path, device, window_size=window_size, overlap_size=overlap_size) # make wav chunks of given length and overlap
                
                spectrograms = audio_to_spectrogram(chunks, sr,device) # make spectrograms
                #predict on spectrograms and save images and data for positive detections
                predictions = predict_and_save_spectrograms(spectrograms, model, device, csv_file_path, wav_file_path, wav_start_time, audio_basename, 
                                                          chunk_start_times, window_size, overlap_size, inverse_label_mapping, time_per_pixel, is_xwav,
                                                          freq_resolution=1, start_freq=10, max_freq=150)
                
                # Write event details and image names to CSV
                # Now write each event detail to the CSV, including the correct image path
                for event in predictions:
                    event['wav_file_path'] = wav_file_path
                    event['model_no'] = model_name
                    writer.writerow(event)
      
print('predictions complete')





















