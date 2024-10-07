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
from AudioStreamDescriptor import WAVhdr, XWAVhdr
import csv
from xwav_functions import get_datetime

# hepler function uses WAVhdr to read wav file header info and extract wav file start time as a datetime object
def extract_wav_start(path):
    
    if path.endswith('.x.wav'):
        xwav_hdr= XWAVhdr(path)
        return xwav_hdr.dtimeStart
    if path.endswith('.wav'):
        wav_hdr = WAVhdr(path)
        return wav_hdr.start
        
   
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
   # chunk_start_times = [i * (window_size - overlap_size) for i in range(len(chunks))]
    chunk_start_times = [start / sr for start in range(0, waveform.shape[1], samples_per_window - samples_overlap)]

    return chunks, sr, chunk_start_times

# Function to convert audio chunks to spectrograms
def audio_to_spectrogram(chunks, sr, device): # these are default fft and hop_length, this is dyamically adjusted depending on the sr. 
    spectrograms = []
   
    for chunk in chunks:
        # Use librosa to compute the spectrogram
        S = torch.stft(chunk[0], n_fft=sr, hop_length=int(sr/10), window=torch.hamming_window(sr).to(device), return_complex=True)
        transform = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80) #convert to dB and clip at 80dB
        S_dB_all = transform(torch.abs(S))
        S_dB = S_dB_all[10:151, :]  # 151 is exclusive, so it includes up to 150 Hz
        spectrograms.append(S_dB.cpu().numpy())
    return spectrograms

      
                
def predict_and_save_spectrograms(spectrograms, model, device, csv_file_path, wav_file_path, wav_start_time, audio_basename, 
                                  chunk_start_times, window_size, overlap_size, inverse_label_mapping, time_per_pixel, is_xwav,
                                  freq_resolution=1, start_freq=10, max_freq=150):
    predictions = []
    csv_base_dir = os.path.dirname(csv_file_path)
    
    # Zip spectrograms and start times
    data = list(zip(spectrograms, chunk_start_times))
    
    for spectrogram_data, chunk_start_time in data:
        
        # Normalize spectrogram and convert to tensor
        normalized_S_dB = (spectrogram_data - np.min(spectrogram_data)) / (np.max(spectrogram_data) - np.min(spectrogram_data))  
        S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L')
        image = ImageOps.flip(S_dB_img)
        # Convert the image to a numpy array for processing
        img_array = np.array(image)
        threshold_1 = 200  # Threshold for the first 10 pixel block
        threshold_2 = 180  # Threshold for the second 10 pixel block
        threshold_3 = 160  # Lower threshold for the third 10 pixel block

        # Gray value to replace the AIS signal
        gray_value = 128  # Mid-gray
        # Find the vertical white lines and gray them out
        # Loop through each column (time slice) in the spectrogram
        for col in range(img_array.shape[1]):  # Loop through each column
        # Check first 10 pixel block (corresponding to the lowest frequency band)
           if np.sum(img_array[-10:, col]) > threshold_1 * 10:
               # If the first 10 pixel block passes, check the second 10 pixel block
               if np.sum(img_array[-20:-10, col]) > threshold_2 * 10:
                   # If the second block passes, check the third block with a lower threshold
                   if np.sum(img_array[-30:-20, col]) > threshold_3 * 10:
                       # If all conditions are met, gray out the entire column
                       img_array[:, col] = gray_value  # Replace the entire column with gray 
        
        # Convert back to image
        final_image = Image.fromarray(img_array)
    
        # Convert to tensor
        S_dB_tensor = F.to_tensor(final_image).unsqueeze(0).to(device)        
        # Run prediction
        model.eval()
        with torch.no_grad():
            prediction = model(S_dB_tensor)
        
        # Extract prediction results
        boxes = prediction[0]['boxes']
        scores = prediction[0]['scores']
        labels = prediction[0]['labels']
       
        # Apply Non-Maximum Suppression (NMS)
        keep_indices = ops.nms(boxes, scores, 0.05)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        # Check if there are valid predictions (boxes)
        if len(boxes) > 0:
            chunk_end_time = chunk_start_time + window_size
            
            # Save the spectrogram image
            image_filename = f"{audio_basename}_second_{int(chunk_start_time)}_to_{int(chunk_end_time)}.png"
            image_path = os.path.join(csv_base_dir, image_filename)
            final_image.save(image_path)
            
            # Iterate through each detection (box)
            for box, score, label in zip(boxes, scores, labels):
                # Convert bounding box coordinates to time and frequency
                # Calculate start and end times from the x-coordinates of the bounding box
                start_time = box[0].item() * time_per_pixel + chunk_start_time
                end_time = box[2].item() * time_per_pixel + chunk_start_time
                # Calculate the lower and upper frequencies from the y-coordinates
                # Assuming y-coordinates are not inverted (higher value for higher frequency)
                lower_freq = (max_freq - box[3].item() * freq_resolution)
                upper_freq = (max_freq - box[1].item() * freq_resolution)
                lower_freq = round(lower_freq)
                upper_freq = round(upper_freq)
                
                if is_xwav:  # distinction between wav and xwav
                    start_datetime = get_datetime(start_time, wav_file_path)
                    end_datetime = get_datetime(end_time, wav_file_path)
                else:
                    start_datetime = wav_start_time + timedelta(seconds=start_time)
                    end_datetime = wav_start_time + timedelta(seconds=end_time)
                # Get textual label from the inverse label mapping
                textual_label = inverse_label_mapping.get(label.item(), 'Unknown')
                
                # Append each detection as a separate row in the predictions list
                predictions.append({
                    'image_file_path': image_path,
                    'label': textual_label,
                    'score': round(score.item(), 2),
                    'start_time_sec': start_time,  # Start time of the detection in the wav file
                    'end_time_sec': end_time,      # End time of the detection in the wav file
                    'start_time': start_datetime,
                    'end_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'min_frequency': round(lower_freq),
                    'max_frequency': round(upper_freq),
                    'box_x1': box[0].item(),
                    'box_x2': box[2].item(),
                    'box_y1': box[1].item(),
                    'box_y2': box[3].item()
                })
    
    return predictions


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










