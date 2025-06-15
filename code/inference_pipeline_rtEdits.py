# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:41:34 2024

@author: Michaela Alksne

script to run the required functions in the correct order to make predictions using trained model
"""

import socket
import struct
import torch
import yaml
import os
import time
import numpy as np
import threading
from datetime import datetime, timedelta
from inference_functions_rt import audio_to_spectrogram, predict_and_save_spectrograms
from call_context_filter import call_context_filter
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

start_time = time.time()

# Load the config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access the configuration variables
wav_file_path = ''
txt_file_path = config['txt_file_path']
model_path = config['model_path']
CalCOFI_flag = config['CalCOFI_flag']
listen_port = config['listen_port']

#Variables for later use in spectrogram generation and filtering
A_thresh=0
B_thresh=0
D_thresh=0
TwentyHz_thresh=0
FourtyHz_thresh=0

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

# Audio params (matched to my simulator (adapted from FreeWilli))
sample_rate = 200000  # e.g. 200 kHz – adjust as needed
bytes_per_sample = 2
channels = 1
samples_per_packet = 200
packet_audio_bytes = samples_per_packet * bytes_per_sample * channels
packet_size = 12 + packet_audio_bytes  # 12 bytes header + audio data
packets_needed = (sample_rate * window_size) // samples_per_packet

timea = time.time()
# Load trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 6 # 5 classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"Model load time: {time.time() - timea:.2f} seconds.")

# --- SHARED RESOURCES ---
audio_buffer = []
#Issues with socket saving to my buffer while inference trying to read from it, prevent that.
stop_event = threading.Event()
buffer_lock = threading.Lock() 
inference_trigger = threading.Event()

# Make a UDP listener to get the packets and save them to the buffer
def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((listen_port, 1045))
    print(f"Listening for UDP packets from {listen_port} on port 1045...")
    eventNumber = 0
    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(packet_size)
            if len(data) != packet_size:
                continue
            audio_data = data[12:]
            if(eventNumber % 1000 == 1):
                print(f"Received packet {eventNumber} at {datetime.utcnow()}")
            with buffer_lock:
                audio_buffer.append(audio_data)
                if len(audio_buffer) >= packets_needed and not inference_trigger.is_set():
                    inference_trigger.set()
                eventNumber += 1
        except socket.error:
            break  # allows clean exit if socket is closed
    sock.close()
    print("No longer listening. Have a whale of a day!")

def inferencer():
    while not stop_event.is_set():
        #Wait for the listener to say there's 60s of data in the buffer
        inference_trigger.wait(timeout=1.0)
        if not inference_trigger.is_set():
            continue  # timeout passed, no trigger

        with buffer_lock:
            if len(audio_buffer) < packets_needed:
                inference_trigger.clear()
                continue  # Not enough data yet
            #Save the bytes I need from the buffer then clear/exit to allow more gathering
            print(f"Received {len(audio_buffer)} packets. Starting inference.")
            full_audio_bytes = b''.join(audio_buffer[:packets_needed])
            del audio_buffer[:packets_needed]
        
        audio_np = np.frombuffer(full_audio_bytes, dtype=np.int16).astype(np.float32)
        audio_tensor = torch.tensor(audio_np).to(device).unsqueeze(0)  # [1, N]

        spectrograms = audio_to_spectrogram(audio_tensor.unsqueeze(0), sample_rate, device)
        spectrogram_data = spectrograms[0]  # now a single spectrogram per call
        window_start_datetime = datetime.utcnow() - timedelta(seconds=window_size)

        predictions = predict_and_save_spectrograms(
            spectrogram_data, model, CalCOFI_flag, device, txt_file_path,
            window_start_datetime, "udp_stream", window_size,
            inverse_label_mapping, time_per_pixel,
            A_thresh, B_thresh, D_thresh, TwentyHz_thresh, FourtyHz_thresh,
            freq_resolution=1, start_freq=10, max_freq=150)

        with open(txt_file_path, mode='a', encoding='utf-8') as txtfile:
            fieldnames = ['wav_file_path', 'model_no', 'image_file_path', 'label', 'score',
                          'start_time_sec', 'end_time_sec', 'start_time', 'end_time',
                          'min_frequency', 'max_frequency', 'box_x1', 'box_x2',
                          'box_y1', 'box_y2']
            for event in predictions:
                event['wav_file_path'] = 'udp_stream'
                event['model_no'] = model_name
                txtfile.write('\t'.join(str(event[f]) for f in fieldnames) + '\n')

        print(f"Inference complete. Processed {len(predictions)} predictions for the last 60 seconds of audio. Output saved to {txt_file_path} and spectrograms saved to folder.")

        

if __name__ == "__main__":
    os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)

    with open(txt_file_path, mode='w', encoding='utf-8') as txtfile:
        txtfile.write('\t'.join([
            'wav_file_path', 'model_no', 'image_file_path', 'label', 'score',
            'start_time_sec', 'end_time_sec', 'start_time', 'end_time',
            'min_frequency', 'max_frequency', 'box_x1', 'box_x2', 'box_y1', 'box_y2'
        ]) + '\n')

    print("Beginning UDP Listener and Inferencer")
    listener_thread = threading.Thread(target=udp_listener, daemon=True)
    inference_thread = threading.Thread(target=inferencer, daemon=True)
    listener_thread.start()
    inference_thread.start()

    try:
        listener_thread.join()
        inference_thread.join()
    except KeyboardInterrupt:
        print("Kill switch activated. Shutting down...")
        stop_event.set()
        listener_thread.join()
        inference_thread.join()
        print("Shutdown complete.")


