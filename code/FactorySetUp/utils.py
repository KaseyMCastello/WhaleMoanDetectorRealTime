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
    # Interpret bytes as big-endian unsigned 16-bit integers
    audio_uint16 = np.frombuffer(audio_bytes, dtype='>u2')  # big-endian uint16 view

    # Convert to int16 without copying: subtract 32768 using int32 view, then cast
    audio_int16 = (audio_uint16.astype(np.int32) - 32768).astype(np.int16)

    # Check for channel alignment
    if audio_uint16.size % num_channels != 0:
        raise ValueError(f"Data length {audio_uint16.size} is not divisible by {num_channels} channels.")

    return audio_int16.reshape(-1, num_channels)

# Function to convert single audio chunk to spectrogram
def audio_to_spectrogram(chunk, sr, device, min_hz, max_hz):
    # Use librosa to compute the spectrogram
    S = torch.stft(chunk, n_fft=sr, hop_length=int(sr/10), window=torch.hamming_window(sr).to(device), return_complex=True)
    transform = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80) #convert to dB and clip at 80dB
    S_dB_all = transform(torch.abs(S))
    S_dB = S_dB_all[0, min_hz:max_hz, :]  # 151 is exclusive, so it includes up to 150 Hz
    return S_dB.cpu().numpy()

   