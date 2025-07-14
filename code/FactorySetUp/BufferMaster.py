import threading
from collections import deque
from datetime import datetime, timedelta
import sys

class BufferMaster:
    """
    Created on Friday, July 11, 2025

    @author: Kasey Castello

    This module defines a BufferMaster class that adds acoustic data packets to a buffer and creates
    instances of Inferencers that process the buffered data. 
    """
    def __init__(self, max_duration_sec, sample_rate, bytes_per_sample, channels):
        '''
        Creates the buffermaster. Requires maximum buffer size (seconds), sample rate (Hz), bytes per sample, and number of channels.
        '''
        self.max_bytes = int(max_duration_sec * sample_rate * bytes_per_sample * channels)
        self.buffer = deque()  # Each item: (timestamp, audio_bytes)
        self.total_bytes = 0
        self.lock = threading.Lock()
        self.inferencers = []
        
        #For packet handling
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.channels = channels
    
    def add_packet(self, audio_bytes: bytes, timestamp: datetime):
        """
        Adds a packet of audio data to the buffer.
        """
        with self.lock:
            #Add the latest packet to the buffer
            self.buffer.append((timestamp, audio_bytes))
            self.total_bytes += len(audio_bytes)

            #If our buffer has gotten too large, remove the oldest packets and send a warning
            #We generally shouldn't get here.
            if (self.total_bytes > self.max_bytes):
                print(f"Warning: Buffer size exceeded {self.max_bytes} bytes at {timestamp}. Removing oldest packets.")
                sys.exit()
                while self.total_bytes > self.max_bytes:
                    _, old_chunk = self.buffer.popleft()
                    self.total_bytes -= len(old_chunk)
        # Trigger inferencers after adding new data
        self.trigger_inferencers()

    def get_audio_window(self, start_time: datetime, duration_ms: float):
        """
            Retrieves exactly the number of bytes corresponding to `duration_ms`,
            starting from `start_time`, even if that means slicing within packets.

            returns audio_bytes
        """
        bytes_needed = int((duration_ms / 1000) * self.sample_rate * self.bytes_per_sample * self.channels)

        collected = bytearray()
        first_packet_found = False
        offset_within_first_packet = 0

        with self.lock:
            for i, (timestamp, data) in enumerate(self.buffer):
                if not first_packet_found:
                    if timestamp > start_time:
                        raise ValueError(f"No packet found with timestamp <= start_time {start_time}")
                    elif timestamp == start_time:
                        offset_within_first_packet = 0
                        first_packet_found = True
                    elif i + 1 < len(self.buffer) and self.buffer[i + 1][0] > start_time:
                        time_diff = (start_time - timestamp).total_seconds()
                        bytes_per_second = self.sample_rate * self.bytes_per_sample * self.channels
                        offset_within_first_packet = int(time_diff * bytes_per_second)
                        if offset_within_first_packet > len(data):
                            raise ValueError("Offset exceeds packet size.")
                        first_packet_found = True
                    else:
                        continue

                if first_packet_found:
                    if offset_within_first_packet:
                        sliced = data[offset_within_first_packet:]
                        collected += sliced
                        offset_within_first_packet = 0
                    else:
                        collected += data

                    if len(collected) >= bytes_needed:
                        return bytes(collected[:bytes_needed])

        raise ValueError("Not enough data in buffer to fulfill request.")
    
    def consume_audio_window(self, start_time: datetime, duration_ms: float):
        """
        Retrieves audio data exactly as in get_audio_window, but also trims the buffer
        up to the end of this window. Only call this from the longest-duration inferencer.
        """
        required_bytes = int((duration_ms / 1000) * self.sample_rate * self.bytes_per_sample * self.channels)
        collected = bytearray()
        first_packet_found = False
        offset_within_first_packet = 0
        end_byte_index = 0  # absolute byte position in buffer
        total_index = 0     # total byte position scanned

        with self.lock:
            packet_indices_to_trim = 0

            for i, (timestamp, data) in enumerate(self.buffer):
                if not first_packet_found:
                    if timestamp > start_time:
                        raise ValueError(f"No packet found with timestamp <= start_time {start_time}")
                    elif timestamp == start_time:
                        offset_within_first_packet = 0
                        first_packet_found = True
                    elif i + 1 < len(self.buffer) and self.buffer[i + 1][0] > start_time:
                        time_diff = (start_time - timestamp).total_seconds()
                        bytes_per_second = self.sample_rate * self.bytes_per_sample * self.channels
                        offset_within_first_packet = int(time_diff * bytes_per_second)
                        if offset_within_first_packet > len(data):
                            raise ValueError("Offset exceeds packet size.")
                        first_packet_found = True
                    else:
                        total_index += len(data)
                        continue

                if first_packet_found:
                    if offset_within_first_packet:
                        sliced = data[offset_within_first_packet:]
                        collected += sliced
                        total_index += len(data)
                        offset_within_first_packet = 0
                    else:
                        collected += data
                        total_index += len(data)

                    if len(collected) >= required_bytes:
                        end_byte_index = total_index - (len(collected) - required_bytes)
                        break

            if len(collected) < required_bytes:
                raise ValueError("Not enough data in buffer to fulfill request.")

            # === Trim buffer up to the byte offset we just finished ===
            bytes_trimmed = 0
            while self.buffer:
                ts, pkt = self.buffer[0]
                if bytes_trimmed + len(pkt) <= end_byte_index:
                    self.total_bytes -= len(pkt)
                    bytes_trimmed += len(pkt)
                    self.buffer.popleft()
                else:
                    # Slice the remaining bytes from the front of the first packet
                    keep_offset = end_byte_index - bytes_trimmed
                    new_data = pkt[keep_offset:]
                    self.total_bytes -= keep_offset
                    self.buffer[0] = (ts, new_data)
                    break

            return bytes(collected[:required_bytes])
        
    def register_inferencer(self, inferencer):
        self.inferencers.append(inferencer)

    def trigger_inferencers(self):
        for inferencer in self.inferencers:
            required_bytes = int( inferencer.duration_ms / 1000 * self.sample_rate * self.bytes_per_sample * self.channels )
            if self.total_bytes >= required_bytes:
                inferencer.trigger()
                



