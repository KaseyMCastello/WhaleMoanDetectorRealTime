# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:41:34 2024

@author: Kasey Castello

edits to Michaela Alksne's code to make inference real time.
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
from inference_functions_rt import audio_to_spectrogram, predict_and_save_spectrograms, convertBackToInt16
from call_context_filter import call_context_filter
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys

start_time = time.time()
udp_timeout = 20 #minutes to wait for a UDP packet before exiting
# Load the config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access the configuration variables
wav_file_path = ''
txt_file_path = config['txt_file_path']
model_path = config['model_path']
CalCOFI_flag = config['CalCOFI_flag']
listen_port = config['listen_port']
sample_rate = config['sample_rate']
samples_per_packet = config['samples_per_packet']
bytes_per_sample = config['bytes_per_sample']
channels = config['channels']
packet_rate = config['packet_rate']

#Variables for later use in spectrogram generation and filtering
A_thresh=0
B_thresh=0
D_thresh=0
TwentyHz_thresh=0
FourtyHz_thresh=0

# Define spectrogram and data parameters
fieldnames = ['wav_file_path', 'model_no', 'image_file_path', 'label', 'score', 'start_time_sec','end_time_sec','start_time','end_time',
              'min_frequency', 'max_frequency','box_x1', 'box_x2', 'box_y1', 'box_y2' ]
# Load the model name from the model path
model_name = os.path.basename(model_path)
visualize_tf = False
label_mapping = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
window_size = 60
overlap_size = 0
time_per_pixel = 0.1  # Since hop_length = sr / 10, this simplifies to 1/10 second per pixel

# Audio params (matched to my simulator (adapted from FreeWilli))
packet_audio_bytes = samples_per_packet * bytes_per_sample * channels
packet_size = 12 + packet_audio_bytes  # 12 bytes header + audio data
bytes_needed = sample_rate * window_size * bytes_per_sample

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
audio_buffer = bytearray()
first_packet_time = None  # To track the first packet time for timestamping
last_packet_time = time.time()  #To track the last packet time for timestamping
last_window_stamp = None #To track the time at which the inference was triggered to be the start of the 60s window
next_window_stamp = None  # To track the next window stamp for inference
last_packet_timestamp = None  # To track the timestamp of the last packet received

#Issues with socket saving to my buffer while inference trying to read from it, prevent that.
stop_event = threading.Event()
buffer_lock = threading.Lock() 
inference_trigger = threading.Event()

# Make a UDP listener to get the packets and save them to the buffer
def udp_listener():
    global first_packet_time  # So we can assign to the shared resource
    global last_packet_time
    global last_window_stamp
    global last_packet_timestamp
    global next_window_stamp

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((listen_port, 5005))
    sock.settimeout(1.0)
    print(f"Listening for UDP packets on port 5005...")
    windowStampSet = False
    EventNumber = 0
    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(packet_size)

            if len(data) != packet_size:
                print(f"Received packet of unexpected size: {len(data)} bytes. Expect: {packet_size} bytes.")
                continue
            last_packet_time = time.time()
            #Infer the timestamp of the packet
            year, month, day, hour, minute, second = struct.unpack("BBBBBB", data[0:6])
            microseconds = int.from_bytes(data[6:10], byteorder='big')
            year += 2000  # Adjust for two-digit format
            last_packet_timestamp = datetime(year, month, day, hour, minute, second, microsecond=microseconds)

            
            # Extract timestamp from first packet
            if first_packet_time is None:
                first_packet_time = last_packet_timestamp
                last_window_stamp = first_packet_time
                print(f"First packet timestamp set to: {first_packet_time}")
            
                    
            audio_data = data[12:]  # 12-byte header; rest is audio
            if(len(audio_data) + len(audio_buffer) == bytes_needed):
                next_window_stamp = last_packet_timestamp
                windowStampSet = True

            with buffer_lock:
                audio_buffer.extend(audio_data)
            if len(audio_buffer) >= bytes_needed and not inference_trigger.is_set():
                print("Buffer has enough data for inference. Triggering inference. Buffer size: ", len(audio_buffer), "Last packet timestamp: ", last_packet_timestamp)
                if (windowStampSet == False ):
                    next_window_stamp = last_window_stamp + timedelta(seconds=window_size)
                inference_trigger.set()
                windowStampSet = False

            if(EventNumber %15000 == 0):
                print(f"Received {len(audio_data)} bytes from packet {EventNumber} at {last_packet_timestamp}. {len(audio_buffer)} bytes in the buffer.")
            EventNumber += 1

        except socket.timeout:
            continue
        except socket.error:
            break  # allows clean exit if socket is closed

    sock.close()
    print("No longer listening. Have a whale of a day!")

def inferencer():
    global last_window_stamp
    global next_window_stamp
    
    while not stop_event.is_set():
        #Wait for the listener to say there's 60s of data in the buffer
        inference_trigger.wait(timeout=1.0)
        if not inference_trigger.is_set():
            continue  # timeout passed, no trigger

        
        
        with buffer_lock: 
            if len(audio_buffer) < bytes_needed:
                inference_trigger.clear()
            else:
                #Save the bytes I need from the buffer then clear/exit to allow more gathering
                full_audio_bytes = audio_buffer[:bytes_needed]
                del audio_buffer[:bytes_needed]

        print(f"Received {bytes_needed} bytes. Starting inference. Window Start Time: {last_window_stamp}, Last Packet Timestamp: {last_packet_timestamp}")
        inference_start_time = time.time()
            
        audio_np = convertBackToInt16(full_audio_bytes, num_channels=1).astype(np.float32)
        
        audio_tensor = torch.tensor(audio_np.squeeze(), dtype=torch.float32).unsqueeze(0).to(device)  # [1, N]
        chunks = []
        chunks.append(audio_tensor)
        
        spectrograms = audio_to_spectrogram(chunks, sample_rate, device)
        
        #spectrogram_data = spectrograms[0]  # now a single spectrogram per call
        chunk_start_timesArray = [last_window_stamp]
        
        predictions =predict_and_save_spectrograms(spectrograms, model, CalCOFI_flag, device, txt_file_path, 
                                      last_window_stamp, "udp_stream", chunk_start_timesArray, window_size, inverse_label_mapping,
                                      time_per_pixel, True, A_thresh, B_thresh, D_thresh, TwentyHz_thresh, FourtyHz_thresh,
                                      freq_resolution=1, start_freq=10, max_freq=150)
        
        with open(txt_file_path, mode='a', encoding='utf-8') as txtfile:
            for event in predictions:
                event['wav_file_path'] = wav_file_path
                event['model_no'] = model_name
                # Write each event as a line in the txt file, tab-separated
                txtfile.write('\t'.join(str(event[field]) for field in fieldnames) + '\n')
        last_window_stamp = next_window_stamp
        print(f"Inference complete. (Took {time.time() - inference_start_time} seconds. Processed {len(predictions)} predictions for the last 60 seconds of audio.")


def timeout_monitor(timeout_seconds=60*udp_timeout):
    while not stop_event.is_set():
        time.sleep(10)  # Check every 10 seconds
        time_since_last_packet = time.time() - last_packet_time
        if time_since_last_packet > timeout_seconds:
            print(f"No packets received in {timeout_seconds / 60:.0f} minutes. Shutting down...")
            stop_event.set()
            break 

if __name__ == "__main__":
    os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)

    with open(txt_file_path, mode='w', encoding='utf-8') as txtfile:
        txtfile.write('\t'.join(fieldnames) + '\n')

    print("Beginning UDP Listener and Inferencer")
    listener_thread = threading.Thread(target=udp_listener, daemon=False)
    inference_thread = threading.Thread(target=inferencer, daemon=False) #Daemon allows the thread to run in the background.
    timeout_thread = threading.Thread(target=timeout_monitor, daemon=True)
    listener_thread.start()
    inference_thread.start()
    timeout_thread.start()

    try:
        listener_thread.join()
        inference_thread.join()
    except KeyboardInterrupt:
        print("Kill switch activated. Shutting down...")
        stop_event.set()
        listener_thread.join()
        inference_thread.join()
        print("Shutdown complete.")


