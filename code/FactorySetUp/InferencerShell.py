"""
Created on Friday, July 11, 2025

@author: Kasey Castello

This module defines an inferencer class that all other inferencers should inherit from. This class shouldn't be used directly.
It provides a basic structure for inferencers to process audio data from a BufferMaster.
"""

import threading
from BufferMaster import BufferMaster
from datetime import datetime, timedelta

class InferencerShell:
    def __init__(self, buffer_master, duration_ms, model_path, sample_rate=200000, bytes_per_sample=2, channels=1):
        """
        Initializes the inferencer with the given parameters.
        """
        self.buffer_master = buffer_master
        self.model_path = model_path
        #For Sample Processing
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.channels = channels
        #For processing audio data
        self.duration_ms = duration_ms
        self.last_inference_time = 0
        self.next_start_time = None
        
        #For executing the inferencer
        self.trigger_event = threading.Event()
        self.stop_event = threading.Event()
        self.buffer_master.register_inferencer(self)

    def print(self):
        print(f"--------------ESTABLISHED {self.name} ------------------")
        print(f"\t MODEL: {self.model_name}")
        print(f"\t MODEL LOAD TIME: {self.model_load_time} s")
        print(f"\t INFERENCE WINDOW: {self.duration_ms /1000} s")
        
    def load_model(self):
        """
        Loads the model needed for inference. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def start(self):
        """
        Starts the inferencer thread.
        """
        threading.Thread(target=self.run, daemon=False).start()
    
    def stop(self):
        """
        Stops the inferencer thread.
        """
        self.stop_event.set()
    
    def trigger(self):
        "Condition for running inference met. Trigger code to run"
        self.trigger_event.set()
    
    def process_audio(self, audio_bytes, start_time):
        """
        Processes a window of audio data. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def run(self):
        """
        Runs the inferencer, processing audio data at specified intervals.
        """
        while not self.stop_event.is_set():
            self.trigger_event.wait()
            self.trigger_event.clear()
            try: 
                if self.next_start_time is None:
                    if not self.buffer_master.buffer:
                        print(f"[{self.name}] Buffer empty when triggered.")
                        continue
                    next_start_time = self.buffer_master.buffer[0][0]

                audio_bytes = self.buffer_master.consume_audio_window( next_start_time, self.duration_ms )

                print(f"[{self.name}] Running inference for window starting {next_start_time}")
                self.process_audio(audio_bytes, next_start_time)

                next_start_time = next_start_time + timedelta(seconds=self.duration_ms / 1000)

            except ValueError as e:
                print(f"[{self.name}] Not enough data: {e}")
        print(f'Inference trigger recieved for {self.name}, killing pipeline')
        