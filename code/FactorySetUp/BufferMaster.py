import threading
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from itertools import islice
import math
import sys
import time
import bisect
from utils import convert_back_to_int16

class RingBuffer:
    """
    Based on reading, I think this will be more efficient that a double-ended queue for getting and adding audio data. 
    
    Ring buffer: pre-allocates array space in a buffer and gives windows of audio data that the buffer can dole to 
    any registered inferencers. **ONLY TO BE DIRECTLY ACCESSED BY THE BM**
    """
    def __init__(self, max_packets: int, packet_audio_bytes: int, packet_duration_ms: float, samples_per_packet, num_channels):
        self.max_packets = max_packets
        self.packet_audio_bytes = packet_audio_bytes
        self.packet_duration_ms = packet_duration_ms

        # Audio buffer: shape (max_packets, packet_audio_bytes)
        self.audio_buffer = np.empty((max_packets, samples_per_packet, num_channels), dtype=np.int16)
        self.timestamps = [None] * max_packets  # Separate timestamp per packet

        self.head = 0  # Next write index
        self.tail = 0  # Oldest valid index
        self.timestamp_index_list = []

        self.lock = threading.Lock()
        self.global_gen = 0
        self.size = 0

        # Active inference windows: set of tuples (start_idx, end_idx, gen)
        self.active_windows = set()
    
    def is_valid(self, idx):
            """
            Check if the id is still inside the ring buffer. (between the tail and head, considering the ring)
            """
            # Prune outdated entries from timestamp_index_list
            tail = self.tail
            head = self.head
            # Return True if idx is currently valid (between tail and head)
            return (tail < head and tail <= idx < head) or (tail > head and (idx >= tail or idx < head)) or (tail == head and self.size == 0)  # empty buffer case
    
    def add_packet(self, audio_bytes: bytes, timestamp: datetime):
        '''
        Add packet to the buffer and update head/tail pointers.
        '''
        audio_int16 = convert_back_to_int16(audio_bytes)

        # Check overwrite safety
        for start_idx, end_idx, gen in list(self.active_windows):
            if start_idx <= self.head <= end_idx or (end_idx < start_idx and (self.head >= start_idx or self.head <= end_idx)):
                print(f"[DEBUG] Overlap detected: head={self.head}, active_window=({start_idx}, {end_idx}, {gen})")
                raise RuntimeError("Buffer overwrite during active inference!")
                
        #Write data to buffer
        with self.lock:
            # Write the data into the audio buffer at 'head' index
            self.audio_buffer[self.head, :len(audio_int16)] = audio_int16
            self.timestamps[self.head] = timestamp
            
            # Update generation counter
            self.global_gen += 1

            # Update timestamp index list for binary searching later
            bisect.insort(self.timestamp_index_list, (timestamp, self.head))

            # Manage ring buffer pointers and size
            if self.size == self.max_packets:
                # Buffer is full, move tail forward to overwrite oldest packet
                self.tail = (self.tail + 1) % self.max_packets
            else:
                # Buffer not full yet, increment size
                self.size += 1

            # Move head forward (circular)
            self.head = (self.head + 1) % self.max_packets
            
            self.timestamp_index_list = [(ts, idx) for ts, idx in self.timestamp_index_list if self.is_valid(idx)]
             
    def get_window(self, start_time: datetime, duration_ms: float):
        """
        Get a view of the window from the start time for duration_ms. 
        NOTE: This method will not return the exact audio window. Instead, it will return a
        view of the packets needed to make the window. So may be up to 1.240 ms extra on the back
        and on the front ends. I did this to minimize buffer lock time. 
        """
        with self.lock:
            msg_time = time.perf_counter() 
            # Binary search for the first packet with timestamp >= start_time
            
            if not self.timestamp_index_list:
                return None, None
            # Check if requested time is too old (data overwritten)
            oldest_time, _ = self.timestamp_index_list[0]
            if start_time < oldest_time:
                return None, None  # data too old / overwritten
            #Binary search for the first packet with timestamp >= start_time
            i = bisect.bisect_left(self.timestamp_index_list, (start_time, 0))
            if i == len(self.timestamp_index_list):
                return None, None  # not enough data
            #If we overshot, back up to last valid timestamp ≤ start_time
            if i > 0 and self.timestamp_index_list[i][0] > start_time:
                i -= 1

            # 5. Extract buffer index from timestamp_index_list
            _, start_idx = self.timestamp_index_list[i]

            packets_needed = math.ceil(duration_ms / self.packet_duration_ms)
            end_idx = (start_idx + packets_needed) % self.max_packets

            self.active_windows.add((start_idx, end_idx, self.global_gen))

            # Extract view
            if start_idx < end_idx:
                view = self.audio_buffer[start_idx:end_idx]
            else:
                view = np.concatenate((self.audio_buffer[start_idx:], self.audio_buffer[:end_idx]), axis=0)
            print(f"Data Capture Time: {(time.perf_counter() - msg_time)*1000} START: {start_idx} END: {end_idx} GEN {self.global_gen} STARTTIME: {start_time}")
            return i, end_idx, self.global_gen, self.timestamps[i], view
        
    def release_window(self, start_idx, end_idx, gen):
        with self.lock:
            self.active_windows.discard((start_idx, end_idx, gen))


class BufferMaster:
    """
    Created on Friday, July 11, 2025

    @author: Kasey Castello

    This module defines a BufferMaster class that adds acoustic data packets to a buffer and triggers
    instances of Inferencers that process the buffered data. 
    """
    def __init__(self, global_stop_event, max_duration_sec, packet_audio_bytes, sample_rate, bytes_per_sample, channels, samples_per_packet):
        '''
        Creates the buffermaster. Requires maximum buffer size (seconds), sample rate (Hz), bytes per sample, and number of channels.
        '''
        #Create the buffer
        self.packet_audio_bytes = packet_audio_bytes
        self.max_packets = int( max_duration_sec * sample_rate * bytes_per_sample * channels / packet_audio_bytes) + 1

        self.total_bytes = 0
        self.lock = threading.Lock()
        self.inferencers = []
        self.stopper = global_stop_event
        
        #For packet handling
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.channels = channels
        self.packet_duration_ms = (packet_audio_bytes / (self.sample_rate * self.bytes_per_sample * self.channels)) * 1000
        
        self.ring_buffer = RingBuffer(max_packets=self.max_packets, packet_audio_bytes=self.packet_audio_bytes, packet_duration_ms=self.packet_duration_ms, 
                                      samples_per_packet=samples_per_packet, num_channels=channels)
    
        self.current_buffer_end = None
        self.current_buffer_start = None

        self.next_window_start = None

        self.timestamps = []    

    def add_packet(self, audio_bytes: bytes, timestamp: datetime):
        """
        Adds a packet of audio data to the buffer.
        """
        try:
            self.ring_buffer.add_packet(audio_bytes, timestamp)
        except RuntimeError as e:
            print(f"Error: {e} — Killing all inferencers due to buffer overwrite")
            self.kill_all()
            self.stopper.set()
            return
        # Trigger inferencers if conditions met
        self.trigger_inferencers()
        
    def get_audio_window(self, start_time: datetime, duration_ms: float):
        """
            Retrieves exactly the number of packets corresponding to `duration_ms`,
            starting from `start_time`. May be a 1.24 ms before start time and after 
            end time. The inferencer will have to handle this.

            returns audio_bytes
        """
        return self.ring_buffer.get_window(start_time, duration_ms)
    
    def release_audio_window(self, start_idx, end_idx, gen):
        self.ring_buffer.release_window(start_idx, end_idx, gen)
        
    def register_inferencer(self, inferencer):
        self.inferencers.append(inferencer)

    def trigger_inferencers(self):
        for inferencer in self.inferencers:
            if inferencer.next_start_time is None and self.ring_buffer.timestamps[self.ring_buffer.tail]:
                inferencer.update_window_start(self.ring_buffer.timestamps[self.ring_buffer.tail])

            if self.ring_buffer.timestamps[self.ring_buffer.tail] is None:
                continue  # buffer empty

            latest_packet_idx = (self.ring_buffer.head - 1) % self.max_packets
            latest_packet_time = self.ring_buffer.timestamps[latest_packet_idx]
            if latest_packet_time is not None and latest_packet_time >= inferencer.next_end_time:
                inferencer.trigger()
                inferencer.update_window_start( inferencer.next_start_time + timedelta(milliseconds=inferencer.duration_ms))
                
    def kill_all(self):
        for prisoner in self.inferencers:
            prisoner.stop()
        self.stopper.set()
        

