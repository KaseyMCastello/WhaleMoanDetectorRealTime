# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:25:24 2024

@author: Michaela ALksne

Make spectrograms and bounding box annotations for each spectrogram in a given wav file
- starts with "modififed annotations" which contain the start and end times of the annotations in seconds since the start of the wav file
- makes "chunks" for each 60 second window in the wav file (with 30 seconds of overlap)
- finds the annotation time stamps for each chunk and makes a spectrogram around them with bounding box annotations in the format:
    [xmin, ymin, xmax, ymax]
- saves the spectrograms and an annotation csv pointing to each spectrogram and its corresponding bounding box
- if multiple bounding boxes exist per spectrogram, the spectrogram annotation gets repeated row-wise
- loops through whatever files you point it to

- puts it on the GPU (5/9/24)
- Add Justin Kim noise reduction (5/21/24)
    
"""

import glob
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from make_spectrograms import generate_spectrogram_and_annotations, generate_spectrogram_and_annotations_PCA

directory_path = "../labeled_data/logs/HARP/modified_annotations" # point to modified annotation files
all_files = glob.glob(os.path.join(directory_path,'*SOCAL26H_modification.csv')) # path for all files

output_directory = "../labeled_data/spectrograms/HARP"
preprocessed_directory = "../labeled_data/spectrograms/HARP/preprocessed"

for file in all_files:
    # Parse the unique part of the filename you want to use for naming
    unique_name_part = Path(file).stem.split('_')[0]  # Adjust index as needed
    annotations_df = pd.read_csv(file)
    # Call your function to process the annotations and generate spectrograms
    generate_spectrogram_and_annotations_PCA(unique_name_part,annotations_df, output_directory, preprocessed_directory, window_size=60, overlap_size=30)



