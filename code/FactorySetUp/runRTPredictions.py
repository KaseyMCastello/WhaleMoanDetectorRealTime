# -*- coding: utf-8 -*-
"""
Created on Friday, July 11, 2025

@author: Kasey Castello

This module combines all of the inferencers and buffer master classes to conduct real-time predictions on 
any input UDP stream
"""

import os
import yaml
import torch
import threading
import struct
import socket
import time
from datetime import datetime, timedelta

from BufferMaster import BufferMaster
from BFWInferencer import BFWInferencer
from Listener import Listener

# === Load config ===
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

wav_file_path = ''
txt_file_path = config['txt_file_path']
model_path = config['model_path']
CalCOFI_flag = config['CalCOFI_flag']
listen_address = config['listen_address']
listen_port = config['listen_port']
sample_rate = config['sample_rate']
samples_per_packet = config['samples_per_packet']
bytes_per_sample = config['bytes_per_sample']
channels = config['channels']
header_size = config['header_size']

packet_audio_bytes = samples_per_packet * bytes_per_sample * channels
packet_size = header_size + packet_audio_bytes  

udp_timeout = 20  # minutes
window_size_sec = 60

# === Make output dir ===
os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
with open(txt_file_path, mode='w', encoding='utf-8') as txtfile:
    txtfile.write('\t'.join([ 'image_path', 'label', 'score','start_time','end_time', 'min_frequency',
                             'max_frequency', 'box_x1','box_x2','box_y1','box_y2']) + '\n')
    

# === Make BufferMaster ===
buffer_master = BufferMaster(
    max_duration_sec = 2 * window_size_sec,
    sample_rate = sample_rate,
    bytes_per_sample = bytes_per_sample,
    channels = channels
)

# === Make Inferencer ===
bfw = BFWInferencer(
    name = "Blue-Fin Whale Inferencer",
    buffer_master = buffer_master,
    duration_ms = window_size_sec * 1000,
    model_path = model_path,
    sample_rate = sample_rate,
    bytes_per_sample = bytes_per_sample,
    channels = channels,
    CalCOFI_flag = CalCOFI_flag,
    output_file_path = txt_file_path,
    file_output_bool = True
)

# === Make Listener ===
listener = Listener(
    listen_address = listen_address,
    listen_port = listen_port,
    packet_size = packet_size,
    buffer_master = buffer_master,
    timeout_duration = udp_timeout
)

# === Start ===
print("---------------------------------------------------------------")
listener.start()
bfw.start()

try:
    listener.stop_event.wait()
except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Shutting down...")
    listener.stop()
    bfw.stop()

print("Shutdown complete.")






