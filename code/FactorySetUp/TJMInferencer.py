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
    def __init__(self, buffer_master, duration_ms, model_path, stop_event, sample_rate=200000, bytes_per_sample=2, channels=1):
        super().__init__( buffer_master, duration_ms, model_path, stop_event, sample_rate, bytes_per_sample, channels )
        self.name = "JOE MODEL (TEST ONLY)"
        self.packetCount = 0
        self.load_model()
        self.print()
    
    def load_model(self):
        self.model_name = "joe_model_onnx_test"
        self.model_load_time = -1
        return
    
    def process_audio(self, audio_bytes, start_time):
            self.packetCount +=1
            if(self.packetCount % 10 == 0):
                 print(f"{self.name} Recieving Packets Appropriately, Current packet count: {self.packetCount}. Window_Time: {self.next_start_time}. Time of Last Packet:  {start_time}.")
                 self.stop_event.set()
        