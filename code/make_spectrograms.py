# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:20:20 2024

@author: Michaela Alksne

Generate noise-reduced, 60 second spectrograms
# default settings input, change 
"""
from pathlib import Path
import torchaudio
import torch
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from time_bbox import time_to_pixels
from freq_bbox import freq_to_pixels
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter

def generate_spectrogram_and_annotations(unique_name_part,annotations_df, output_dir, window_size=60, overlap_size=30, n_fft=48000, hop_length=4800):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_list = []  # To collect annotations for each deployment

    # Group annotations by audio file
    grouped = annotations_df.groupby('audio_file')

    for audio_file_path, group in grouped:
        audio_path = Path(audio_file_path)  # Convert to Path object for easier handling
        audio_basename = audio_path.stem  # Get the basename without the extension
        print(audio_basename)
        # Load the audio file
        waveform, sr = torchaudio.load(audio_file_path)
        waveform = waveform.to('cuda') # move wavform to gpu for efficent processing
        samples_per_window = window_size * sr
        samples_overlap = overlap_size * sr

        # Process each chunk of the audio file
        for start_idx in range(0, waveform.shape[1], samples_per_window - samples_overlap):
            end_idx = start_idx + samples_per_window
            if end_idx > waveform.shape[1]:
                # If the remaining data is less than the window size, pad it with zeros
                padding_size = end_idx - waveform.shape[1]
                chunk = torch.nn.functional.pad(waveform[:, start_idx:], (0, padding_size))  # Pad the last part of the waveform
            else:
                chunk = waveform[:, start_idx:end_idx]
            # Compute STFT on GPU
            S = torch.stft(chunk[0], n_fft=sr, hop_length=int(sr/10), window=torch.hamming_window(sr).to('cuda'), return_complex=True)
            S_dB_all = torchaudio.transforms.AmplitudeToDB()(torch.abs(S))
            
            S_dB = S_dB_all[10:151, :]  # 151 is exclusive, so it includes up to 150 Hz

            chunk_start_time = start_idx / sr
            chunk_end_time = chunk_start_time + window_size
            
            spectrogram_filename = f"{audio_basename}_second_{int(chunk_start_time)}_to_{int(chunk_end_time)}.png"
            spectrogram_data = S_dB.cpu().numpy()  # Move data to CPU for image processing
            # Filter and adjust annotations for this chunk
            relevant_annotations = group[(group['start_time'] >= chunk_start_time) & (group['end_time'] <= chunk_end_time)]

            # Adjust annotation times relative to the start of the chunk
            for _, row in relevant_annotations.iterrows():
                adjusted_start_time = row['start_time'] - chunk_start_time
                adjusted_end_time = row['end_time'] - chunk_start_time
                
                # Perform column-wise background subtraction
                for j in range(spectrogram_data.shape[1]):
                    column = spectrogram_data[:, j]
                    percentile_value = np.percentile(column, 60)
                    # Subtract the percentile value from each column and clip to ensure no negative values
                    spectrogram_data[:, j] = np.clip(column - percentile_value, 0, None)

                # Raise the modified spectrogram data to the power of 6 to enhance contrast
                enhanced_image = np.power(spectrogram_data,3)
                
                # Normalize the results to make sure they fit in the 0-255 range for image conversion
                enhanced_image = 255 * (enhanced_image / enhanced_image.max())

                # Convert the processed data back to an image
                final_image = Image.fromarray(enhanced_image.astype(np.uint8), 'L')
    
                # Flip the image vertically
                final_image = ImageOps.flip(final_image)

                final_image.save(output_dir / spectrogram_filename)
              
                #S_dB_img.save(output_dir / spectrogram_filename)
                # Map annotation times and frequencies to spectrogram pixels
                xmin, xmax = time_to_pixels(adjusted_start_time, adjusted_end_time, S_dB.shape[1], window_size)
                ymin, ymax = freq_to_pixels(row['low_f'], row['high_f'], S_dB.shape[0], sr, sr)

                annotations_list.append({
                    "spectrogram_path": f"{output_dir}\{spectrogram_filename}",
                    "label": row['annotation'],
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
                
                
    # Convert annotations list to DataFrame and save as CSV
    df_annotations = pd.DataFrame(annotations_list)
    df_annotations.to_csv(f"{output_dir}/{unique_name_part}_annotations.csv", index=False)
    
    
# Justin Kim version of the code where he integrates PCA and noise reduction as preprocessing step. Saves original spectrograms and preprocessed/noise reduced ones.     
    
def generate_spectrogram_and_annotations_PCA(unique_name_part,annotations_df, output_dir, preprocessed_directory, window_size=60, overlap_size=30):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_list = []  # To collect annotations for each deployment

    # Group annotations by audio file
    grouped = annotations_df.groupby('audio_file')

    for audio_file_path, group in grouped:
        audio_path = Path(audio_file_path)  # Convert to Path object for easier handling
        audio_basename = audio_path.stem  # Get the basename without the extension
        print(audio_basename)
    
        # Load the audio file
        waveform, sr = torchaudio.load(audio_file_path)
        waveform = waveform.to('cuda') # move wavform to gpu for efficent processing
        samples_per_window = window_size * sr
        samples_overlap = overlap_size * sr

        # Process each chunk of the audio file
        for start_idx in range(0, waveform.shape[1], samples_per_window - samples_overlap):
            end_idx = start_idx + samples_per_window
            if end_idx > waveform.shape[1]:
                # If the remaining data is less than the window size, pad it with zeros
                padding_size = end_idx - waveform.shape[1]
                chunk = torch.nn.functional.pad(waveform[:, start_idx:], (0, padding_size))  # Pad the last part of the waveform
            else:
                chunk = waveform[:, start_idx:end_idx]
            # Compute STFT on GPU
            S = torch.stft(chunk[0], n_fft=sr, hop_length=int(sr/10), window=torch.hamming_window(sr).to('cuda'), return_complex=True)
            S_dB_all = torchaudio.transforms.AmplitudeToDB()(torch.abs(S))
            
            S_dB = S_dB_all[10:151, :]  # 151 is exclusive, so it includes up to 150 Hz

            chunk_start_time = start_idx / sr
            chunk_end_time = chunk_start_time + window_size
            
            spectrogram_filename = f"{audio_basename}_second_{int(chunk_start_time)}_to_{int(chunk_end_time)}.png"
            spectrogram_data = S_dB.cpu().numpy()  # Move data to CPU for image processing


            # Filter and adjust annotations for this chunk
            relevant_annotations = group[(group['start_time'] >= chunk_start_time) & (group['end_time'] <= chunk_end_time)]

            # Adjust annotation times relative to the start of the chunk
            for _, row in relevant_annotations.iterrows():
                adjusted_start_time = row['start_time'] - chunk_start_time
                adjusted_end_time = row['end_time'] - chunk_start_time
                
                normalized_S_dB = (spectrogram_data - np.min(spectrogram_data)) / (np.max(spectrogram_data) - np.min(spectrogram_data))
                S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L') # apply grayscale colormap
                # Convert the processed data back to an image
    
                # Flip the image vertically
                final_image = ImageOps.flip(S_dB_img)

                final_image.save(output_dir / spectrogram_filename)
              
                #S_dB_img.save(output_dir / spectrogram_filename)
                # Map annotation times and frequencies to spectrogram pixels
                xmin, xmax = time_to_pixels(adjusted_start_time, adjusted_end_time, S_dB.shape[1], window_size)
                ymin, ymax = freq_to_pixels(row['low_f'], row['high_f'], S_dB.shape[0], sr, sr)

                annotations_list.append({
                    "spectrogram_path": f"{output_dir}\{spectrogram_filename}",
                    "label": row['annotation'],
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    # Convert annotations list to DataFrame and save as CSV
    df_annotations = pd.DataFrame(annotations_list)
    df_annotations.to_csv(f"{output_dir}/{unique_name_part}_annotations.csv", index=False)

    # Read the annotations CSV
    annotations = pd.read_csv(f'{output_dir}/{unique_name_part}_annotations.csv')
    unique_annotation = annotations.drop_duplicates(subset=['spectrogram_path'])
    base_folder_name = os.path.basename(os.path.normpath(preprocessed_directory))
    annotations_modded = annotations.copy()
    
    # Modify the 'spectrogram_path' column to add "preprocessed" as a subfolder
    annotations_modded["spectrogram_path"] = annotations["spectrogram_path"].apply(
        lambda path: os.path.join(os.path.dirname(path), 'preprocessed', os.path.basename(path))
    )
    
    # Save the modified annotations to a new CSV file
    annotations_modded.to_csv(f"{preprocessed_directory}/{unique_name_part}_preprocessed_annotations.csv", index=False)

    
    # Initialize the data matrix
    data_matrix = []
    
    # Read and process each spectrogram image
    for index, row in annotations.iterrows():
        image = Image.open(row['spectrogram_path'])
        print(row['spectrogram_path'])
        pixel_values = np.array(image).flatten()
        data_matrix.append(pixel_values)
    
    # Stack the data matrix
    stacked_specs = np.vstack(data_matrix)
    
    # Standardize the data
    scaler = StandardScaler(with_std=False)
    data_matrix_mod1 = scaler.fit_transform(stacked_specs)
    original_data = data_matrix_mod1
    
    # Perform SVD
    U, S, T = np.linalg.svd(original_data, full_matrices=False)
    
    # Multiply U and S
    US = U @ np.diag(S)
    
    # Check shapes before matrix multiplication
    print(f"Shape of US: {US.shape}")
    print(f"Shape of T: {T.shape}")
    
    # Truncate T to the desired shape (150 rows)
    T = T[:150, :]
    
    # Initialize the signal enhanced features matrix
    signal_enhanced_features = np.zeros_like(T)
    
    # Process T to extract features
    for i in range(len(T)):
        feature = np.copy(T[i].reshape((141, 601)))
        for j in range(feature.shape[1]):
            column = feature[:, j]
            percentile_value = np.percentile(column, 20)
            feature[:, j] = column - percentile_value
            feature[:, j][feature[:, j] < 0] = 0
        signal_enhanced_features[i] = feature.flatten()
    
    # Check shapes before final matrix multiplication
    print(f"Shape of US[:, :150]: {US[:, :150].shape}")
    print(f"Shape of signal_enhanced_features: {signal_enhanced_features.shape}")
    
    # Perform matrix multiplication
    matrix = US[:, :150] @ signal_enhanced_features
    
    # Scale back the matrix
    matrix_scaled = scaler.inverse_transform(matrix)
    matrix_scaled = np.where(matrix_scaled < 0, 0, matrix_scaled)
    
    # Initialize the matrix for saving processed images
    matr_sub = np.zeros_like(matrix_scaled)
    
    # Process and filter each spectrogram
    for i in range(len(matrix_scaled)):
        spec = np.copy(matrix_scaled[i].reshape((141, 601)))
        for j in range(spec.shape[1]):
            column = spec[:, j]
            percentile_value = np.percentile(column, 60)
            spec[:, j] = column - percentile_value
            spec[:, j][spec[:, j] < 0] = 0
        spec = median_filter(spec, size=3)
        matr_sub[i] = spec.flatten()
    
    # Extract filenames for saving processed images
    filenames = annotations_modded["spectrogram_path"].apply(lambda x: os.path.basename(x)).reset_index(drop=True)
    print(filenames)
    
    # Save each processed image
    for i in range(len(matr_sub)):
        processed_image = matr_sub[i].reshape(141, 601)
        image = Image.fromarray(processed_image.astype(np.uint8), 'L')
        image.save(Path(preprocessed_directory) / Path(filenames[i]))




 
