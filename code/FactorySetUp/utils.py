"""
Created on Friday, July 11, 2025

@author: Kasey Castello

This module defines useful functions that could be used by any inferencer
"""
import os
import torch
import torchaudio
import torchvision
import struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

def convert_back_to_int16(audio_bytes, num_channels=1):
    # Unpack bytes to uint16
    num_samples = len(audio_bytes) // 2  # 2 bytes per uint16
    audio_uint16 = struct.unpack(f'>{num_samples}H', audio_bytes)
    audio_uint16 = np.array(audio_uint16, dtype=np.uint16)

    # Convert to int16 by reversing +32768 offset
    audio_int16 = (audio_uint16.astype(np.int32) - 32768).astype(np.int16)

    # Reshape into (samples, channels)
    if num_samples % num_channels != 0:
        raise ValueError(f"Data length {num_samples} is not divisible by {num_channels} channels.")

    audio_int16 = audio_int16.reshape(-1, num_channels)
    return audio_int16

# Function to convert single audio chunk to spectrogram
def audio_to_spectrogram(chunk, sr, device, min_hz, max_hz):
    # Use librosa to compute the spectrogram
    S = torch.stft(chunk, n_fft=sr, hop_length=int(sr/10), window=torch.hamming_window(sr).to(device), return_complex=True)
    transform = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80) #convert to dB and clip at 80dB
    S_dB_all = transform(torch.abs(S))
    S_dB = S_dB_all[0, min_hz:max_hz, :]  # 151 is exclusive, so it includes up to 150 Hz
    return S_dB.cpu().numpy()

   