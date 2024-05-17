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
from AudioStreamDescriptor import WAVhdr
from modify_timestamp_function import modify_annotations
import random
import pandas as pd
import numpy as np

directory_path = "../labeled_data/logs/HARP" # point to original logger files
all_files = glob.glob(os.path.join(directory_path,'*.xls')) # path for all files

new_base_path = '../labeled_data/wav' # path to change to 

# make a subfolder for saving modified logs 
subfolder_name = "modified_annotations"
# Create the subfolder if it doesn't exist
subfolder_path = os.path.join(directory_path, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True)

# loop through all annotation files and save them in subfolder "modified_annotations"

for file in all_files:
    data = pd.read_excel(file)
    subset_df = modify_annotations(data, new_base_path)
    filename = os.path.basename(file)
    new_filename = filename.replace('.xls', '_modification.csv')
     # Construct the path to save the modified DataFrame as a CSV file
    save_path = os.path.join(subfolder_path, new_filename)
    # Save the subset DataFrame to the subset folder as a CSV file
    subset_df.to_csv(save_path, index=False)


