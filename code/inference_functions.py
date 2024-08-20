# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:41:34 2024

@author: Michaela Alksne

All of the functions needed to run inference data processing pipeline...
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
import torchvision.ops as ops
from PIL import ImageOps
from datetime import datetime, timedelta
from IPython.display import display
import torchaudio
from AudioStreamDescriptor import WAVhdr
import csv

# hepler function uses WAVhdr to read wav file header info and extract wav file start time as a datetime object
def extract_wav_start(path):
    wav_hdr = WAVhdr(path)
    wav_start_time = wav_hdr.start
    return wav_start_time
# Function to load audio file and chunk it into overlapping windows

def chunk_audio(audio_file_path, device, window_size=60, overlap_size=5):
    # Load audio file
    waveform, sr = torchaudio.load(audio_file_path)
    waveform = waveform.to(device)  # Move waveform to GPU for efficient processing
    samples_per_window = window_size * sr
    samples_overlap = overlap_size * sr

    # Calculate the number of chunks
    chunks = []
    for start in range(0, waveform.shape[1], samples_per_window - samples_overlap):
        end = start + samples_per_window
        # If the last chunk is smaller than the window size, pad it with zeros
        if end > waveform.shape[1]:
            y_pad = torch.nn.functional.pad(waveform[:, start:], (0, end - waveform.shape[1]), mode='constant')
            chunks.append(y_pad)
        else:
            chunks.append(waveform[:, start:end])
    
    return chunks, sr

# Function to convert audio chunks to spectrograms
def audio_to_spectrogram(chunks, sr, device): # these are default fft and hop_length, this is dyamically adjusted depending on the sr. 
    spectrograms = []
   
    for chunk in chunks:
        # Use librosa to compute the spectrogram
        S = torch.stft(chunk[0], n_fft=sr, hop_length=int(sr/10), window=torch.hamming_window(sr).to(device), return_complex=True)
        S_dB_all = torchaudio.transforms.AmplitudeToDB()(torch.abs(S))
        # Convert to dB
        S_dB = S_dB_all[10:151, :]  # 151 is exclusive, so it includes up to 150 Hz
        spectrograms.append(S_dB.cpu().numpy())
    return spectrograms

        
def predict_and_plot_on_spectrograms(spectrograms, model, device, visualize = True):
    predictions = []
    font = ImageFont.truetype("arial.ttf", 16)  # Adjust the font and size as needed
    
    for spectrogram_data  in spectrograms:
        
        normalized_S_dB = (spectrogram_data - np.min(spectrogram_data)) / (np.max(spectrogram_data) - np.min(spectrogram_data)) # normalize spectrogram 
        #enhanced_image = np.power(normalized_S_dB,3)
        S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L') # apply grayscale colormap
        # Flip the image vertically
        final_image = ImageOps.flip(S_dB_img)
        S_dB_tensor = F.to_tensor(final_image).unsqueeze(0)  # Add batch dimension
        S_dB_tensor=S_dB_tensor.to(device)
        
        # Run prediction
        model.eval()
        with torch.no_grad():
            prediction = model(S_dB_tensor)
        predictions.append(prediction)
        
        
        if visualize: 
            draw = ImageDraw.Draw(final_image)
            
            # Assuming `prediction` contains `boxes`, `labels`, and `scores`
            boxes = prediction[0]['boxes']
            scores = prediction[0]['scores']
            labels = prediction[0]['labels']
            
            # Apply Non-Maximum Suppression (NMS) for cleaner visualization
            keep_indices = ops.nms(boxes, scores, 0.2)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
            
            # Draw each bounding box and label on the spectrogram image
            for box, score, label in zip(boxes, scores, labels):
                score_formatted = round(score.item(), 2)
                # Convert box coordinates (considering the flip if necessary)
                box = box.tolist()
                draw.rectangle(box, outline="white")
                draw.text((box[0], box[1]-20), f"Label: {label}, Score: {score_formatted}", fill="white", font=font)
            
            # Display the spectrogram with drawn predictions
           #S_dB_img.show()
            display(final_image)
        else:
           
            pass
            
    return predictions
        

# filter predictions based on defined parameters 

def apply_filters_to_predictions(predictions, nms_threshold=0.2, D_threshold=0, fortyHz_threshold=0, 
                                      twentyHz_threshold=0, A_threshold=0, B_threshold=0):
                                    
    """
    Apply NMS on the predictions to filter out overlapping bounding boxes, then filter
    each category by specific score thresholds. 
    Args:
        predictions (list): A list of predictions where each prediction is a dict
            containing 'boxes', 'labels', and 'scores'.
        nms_threshold (float): The NMS IoU threshold.
        [category]_threshold (float): The score threshold for each call category.
       
        
    Returns:
        list: A list of filtered predictions.
    """
    filtered_predictions = []
    # Time per pixel in the spectrogram
    time_per_pixel = 0.1 # 100 miliseconds or 1/10 of a second for sr = x and hop_length = x/10
    
    # Assuming labels are integers and mapping them accordingly
    label_mapping = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}
    thresholds = {
        label_mapping['D']: D_threshold,
        label_mapping['40Hz']: fortyHz_threshold,
        label_mapping['20Hz']: twentyHz_threshold,
        label_mapping['A']: A_threshold,
        label_mapping['B']: B_threshold
    }

    for prediction in predictions:
        boxes = prediction[0]['boxes']
        scores = prediction[0]['scores']
        labels = prediction[0]['labels']

        # Apply NMS
        keep_indices = ops.nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

        final_boxes, final_scores, final_labels = [], [], []

        for box, score, label in zip(boxes, scores, labels):
            if score < thresholds[label.item()]:
                continue # skip this prediction
            
           # if label.item() == label_mapping['D']: # convert D calls to 40 Hz if they are less than 1.5 seconds in duration... 
                # Calculate duration of the D call
            #    duration_seconds = (box[2] - box[0]).item() * time_per_pixel
             #   if duration_seconds < 1.5:
                    # Convert to 40Hz call
               #     label = torch.tensor(label_mapping['40Hz'])
                    
         #   if label.item() == label_mapping['20Hz']: # High scoring 20 Hz that are within the bounds of 40 Hz should be converted to 40 Hz. 
          #     if  box[1] > 40 and box[3] < 100: 
               #    label = torch.tensor(label_mapping['40Hz']) # convert to 40 Hz.
          #     elif box[1] > 40 and box[3] > 100:  
          #         continue
                   
         #   if label.item() == label_mapping['A']:
          #     if box[1] < 50 : # remove false positive A calls with ymin less than 60 Hz
          #         continue #skip this prediction. 
         #   if label.item() == label_mapping['B']:
          #      if box[3] > 60: # remove false positive B calls with ymax greater than 70 Hz
           #         continue # skip this prediction
               
            final_boxes.append(box)
            final_scores.append(score)
            final_labels.append(label)

        filtered_predictions.append([{
            'boxes': torch.stack(final_boxes) if final_boxes else torch.tensor([]),
            'labels': torch.stack(final_labels) if final_labels else torch.tensor([]),
            'scores': torch.stack(final_scores) if final_scores else torch.tensor([]),
        }])

    return filtered_predictions


def save_filtered_images(spectrograms, filtered_predictions, csv_file_path, audio_basename, chunk_start_times, window_size, overlap_size):
    # Use the base directory of the CSV file path to place the "images" directory
    csv_base_dir = os.path.dirname(csv_file_path)
    
    saved_image_paths = []

    for index, prediction_list in enumerate(filtered_predictions):
        prediction = prediction_list[0]

        # Proceed with plotting and saving only if there are detections that passed filtering
        if len(prediction['boxes']) > 0:
            S_dB = spectrograms[index]
                
            
            normalized_S_dB = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB)) # normalize spectrogram 
           # enhanced_image = np.power(normalized_S_dB,3)
            S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L') # apply grayscale colormap
            # Flip the image vertically
            final_image = ImageOps.flip(S_dB_img)
            
            chunk_start_time = chunk_start_times[index]
            chunk_end_time = chunk_start_time + window_size
            
            # The image filename now includes the audio basename
            image_filename = f"{audio_basename}_second_{int(chunk_start_time)}_to_{int(chunk_end_time)}.png"
            image_path = os.path.join(csv_base_dir, image_filename)
            final_image.save(image_path)
            saved_image_paths.append(image_path) #save the image
                
    return saved_image_paths


# convert bounding box x and y coordinates to timestamp and frequency within the wav file
def bounding_box_to_time_and_frequency(box, chunk_index, chunk_start_times, time_per_pixel, freq_resolution=1, start_freq=10, max_freq=150):
    # Calculate start and end times from the x-coordinates of the bounding box
    box_x1, box_x2 = box[0].item(), box[2].item()
    start_time = box_x1 * time_per_pixel + chunk_start_times[chunk_index]
    end_time = box_x2 * time_per_pixel + chunk_start_times[chunk_index]
    
    # Calculate the lower and upper frequencies from the y-coordinates
    # Assuming y-coordinates are not inverted (higher value for higher frequency)
    box_y1, box_y2 = box[1].item(), box[3].item()
        
    # Invert the y-axis to correctly map the frequencies
    lower_freq = (max_freq - box_y2 * freq_resolution) 
    upper_freq = (max_freq - box_y1 * freq_resolution)
  
    lower_freq = round(lower_freq)
    upper_freq = round(upper_freq)
    
    return start_time, end_time, lower_freq, upper_freq, box_x1, box_x2, box_y1, box_y2

def predictions_to_datetimes_frequencies_and_labels(filtered_predictions, chunk_start_times, time_per_pixel, wav_start_datetime, freq_resolution=1, start_freq=10):
    results = []
    label_mapping = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5} # why did I not call this by the logger conventions? because I wanted to make it even more difficult for myself. sick one. 
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}

    for chunk_index, prediction in enumerate(filtered_predictions):
        for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
            start_time, end_time, lower_freq, upper_freq, box_x1, box_x2, box_y1, box_y2 = bounding_box_to_time_and_frequency(box, chunk_index, chunk_start_times, time_per_pixel, freq_resolution, start_freq)
            start_datetime = wav_start_datetime + timedelta(seconds=start_time)
            end_datetime = wav_start_datetime + timedelta(seconds=end_time)
            textual_label = inverse_label_mapping[label.item()]

            results.append({
                'label': textual_label,
                'score': round(score.item(), 2),  # Round the score for readability if desired
                'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],  # Format as a string
                'start_time_sec': start_time,
                'end_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'end_time_sec': end_time,
                'min_frequency': round(lower_freq),  # Round frequency to the nearest integer
                'max_frequency': round(upper_freq), 
                'box_x1': box_x1, 
                'box_x2': box_x2,
                'box_y1': box_y1, 
                'box_y2': box_y2
            
            })
    return results


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
    
    # Calculate the 75th percentile for the signal values (upper 25th %)
    signal_75th_percentile = np.percentile(cropped_array, 75)
    # Calculate the 25th percentile for the noise values (lower 25th %)
    noise_25th_percentile = np.percentile(cropped_image, 25)
    
    # Check if the noise percentile is zero, and set it to 1
    # doing this because when we divide by 1, it doesnt change affect the snr value and we just get back the relationship bt difference in signal and noise... i suppose
    if noise_25th_percentile == 0:
        noise_25th_percentile = 1

    # Compute SNR
    snr = signal_75th_percentile / noise_25th_percentile

    return snr










