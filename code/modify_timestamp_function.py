# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:11:22 2024

@author: Michaela Alksne
"""

import os
from extract_wav_header import extract_wav_start 

# helper function to modify the original logger files
def modify_annotations(df, new_base_path):
    
    df['audio_file'] = [in_file.replace(os.path.split(in_file)[0], new_base_path) for in_file in df['Input file']] # uses list comprehension to replace old wav path with new one
    df['file_datetime']=df['audio_file'].apply(extract_wav_start) # uses .apply to apply extract_wav_start time from each wav file in the list
    df['start_time'] = (df['Start time'] - df['file_datetime']).dt.total_seconds() # convert start time difference to total seconds
    df['end_time'] = (df['End time'] - df['file_datetime']).dt.total_seconds() # convert end time difference to total seconds
    df['annotation']= df['Call']
    df['high_f'] = df['Parameter 1']
    df['low_f'] = df['Parameter 2']
    df = df.loc[:, ['audio_file','annotation','high_f','low_f','start_time','end_time']] # subset all rows by certain column name
    
    return df