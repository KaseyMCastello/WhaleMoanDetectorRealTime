"""
Created on Friday, July 11, 2025

@author: Kasey Castello

Will delete this model, currently for testing of two-model inference.
"""
import numpy as np
import pickle
import sklearn
import torch

from InferencerShell import InferencerShell

class TestJoeModelInferencer(InferencerShell):
    """Class to detect/classify odontocete clicks."""
    def __init__( buffer_master, duration_ms, model_path, sample_rate=200000, bytes_per_sample=2, channels=1):
        self.name = "Joe's Click-Classifier (TEST)"
        self.cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
    
    def load_model(self):
        return
    
    def process_audio(self, audio_bytes, start_time):
        return
