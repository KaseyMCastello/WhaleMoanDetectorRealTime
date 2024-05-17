# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:35:46 2024

@author: Michaela Alksne 

Script to run when modifying Triton logger annotation excel datasheets
converts xls to csv containing the audio file path, the annotation label, the frequency bounds, and time bounds. 
saves new csv in "modified annotations subfolder"
wav is the audio file
start time = start time of call in number of seconds since start of wav 
end time = end time of call in number of seconds since start of wav

"""

from datetime import datetime
import os
import glob
import sys
#from opensoundscape import Audio, Spectrogram
sys.path.append(r"L:\Sonobuoy_faster-rCNN_MNA_PCA\code\PYTHON")
from AudioStreamDescriptor import WAVhdr
import random
import pandas as pd
import numpy as np

directory_path = "L:\\Sonobuoy_faster-rCNN_MNA_PCA\\labeled_data\\logs\\HARP" # point to original logger files
all_files = glob.glob(os.path.join(directory_path,'*.xls')) # path for all files

new_base_path = 'L:\\Sonobuoy_faster-rCNN_MNA_PCA\\labeled_data\\wav' # path to change to 

# hepler function uses WAVhdr to read wav file header info and extract wav file start time as a datetime object
def extract_wav_start(path):
    wav_hdr = WAVhdr(path)
    wav_start_time = wav_hdr.start
    return wav_start_time

# helper function to modify the original logger files
def modify_annotations(df):
    
    df['audio_file'] = [in_file.replace(os.path.split(in_file)[0], new_base_path) for in_file in df['Input file']] # uses list comprehension to replace old wav path with new one
    df['file_datetime']=df['audio_file'].apply(extract_wav_start) # uses .apply to apply extract_wav_start time from each wav file in the list
    df['start_time'] = (df['Start time'] - df['file_datetime']).dt.total_seconds() # convert start time difference to total seconds
    df['end_time'] = (df['End time'] - df['file_datetime']).dt.total_seconds() # convert end time difference to total seconds
    df['annotation']= df['Call']
    df['high_f'] = df['Parameter 1']
    df['low_f'] = df['Parameter 2']
    df = df.loc[:, ['audio_file','annotation','high_f','low_f','start_time','end_time']] # subset all rows by certain column name
    
    return df
    
# make a subfolder for saving modified logs 
subfolder_name = "modified_annotations"
# Create the subfolder if it doesn't exist
subfolder_path = os.path.join(directory_path, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True)

# loop through all annotation files and save them in subfolder "modified_annotations"

for file in all_files:
    data = pd.read_excel(file)
    subset_df = modify_annotations(data)
    filename = os.path.basename(file)
    new_filename = filename.replace('.xls', '_modification.csv')
     # Construct the path to save the modified DataFrame as a CSV file
    save_path = os.path.join(subfolder_path, new_filename)
    # Save the subset DataFrame to the subset folder as a CSV file
    subset_df.to_csv(save_path, index=False)


