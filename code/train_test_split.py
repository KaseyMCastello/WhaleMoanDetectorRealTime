# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:42:59 2024

@author: Michaela Alksne

make train and test splits
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

save_path = '../figures' # image save path

# Read in CalCOFI the slow way so I can print everything

CalCOFI_2004_07 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC200407_annotations.csv')
print(f'CalCOFI_2004_07: {CalCOFI_2004_07["label"].value_counts()}')

CalCOFI_2006_02 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC200602_annotations.csv')
print(f'CalCOFI_2006_02: {CalCOFI_2006_02["label"].value_counts()}')

CalCOFI_2006_07 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC200602_annotations.csv')
print(f'CalCOFI_2006_07: {CalCOFI_2006_07["label"].value_counts()}')

CalCOFI_2007_07 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC200707_annotations.csv')
print(f'CalCOFI_2007_07: {CalCOFI_2007_07["label"].value_counts()}')

CalCOFI_2007_11 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC200711_annotations.csv')
print(f'CalCOFI_2007_11: {CalCOFI_2007_11["label"].value_counts()}')

CalCOFI_2008_08 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC200808_annotations.csv')
print(f'CalCOFI_2008_08: {CalCOFI_2008_08["label"].value_counts()}')

CalCOFI_2011_04 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC201104_annotations.csv')
print(f'CalCOFI_2011_04: {CalCOFI_2011_04["label"].value_counts()}')

CalCOFI_2011_08 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC201108_annotations.csv')
print(f'CalCOFI_2011_08: {CalCOFI_2011_08["label"].value_counts()}')

CalCOFI_2011_11 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC201111_annotations.csv')
print(f'CalCOFI_2011_11: {CalCOFI_2011_11["label"].value_counts()}')

CalCOFI_2015_01 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC201501_annotations.csv')
print(f'CalCOFI_2015_01: {CalCOFI_2015_01["label"].value_counts()}')

CalCOFI_2017_01 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC201701_annotations.csv')
print(f'CalCOFI_2017_01: {CalCOFI_2017_01["label"].value_counts()}')

CalCOFI_2017_08 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC201708_annotations.csv')
print(f'CalCOFI_2017_08: {CalCOFI_2017_08["label"].value_counts()}')

CalCOFI_2019_07 = pd.read_csv('../labeled_data/spectrograms/CalCOFI/CC201907_annotations.csv')
print(f'CalCOFI_2019_07: {CalCOFI_2019_07["label"].value_counts()}')

# List of all DataFrames
CalCOFI_train_df = [
    CalCOFI_2004_07, CalCOFI_2006_02, CalCOFI_2006_07, CalCOFI_2007_07,
    CalCOFI_2007_11, CalCOFI_2011_04, CalCOFI_2011_08,
    CalCOFI_2011_11, CalCOFI_2015_01, CalCOFI_2017_01, CalCOFI_2017_08,
    CalCOFI_2019_07]


# Concatenate all DataFrames into one
CalCOFI_train = pd.concat(CalCOFI_train_df, ignore_index=True)

print(f'CalCOFI traning data:{CalCOFI_train["label"].value_counts()}')

#read in HARP slow way too

DCPP01A = pd.read_csv('../labeled_data/spectrograms/HARP/DCPP01A_annotations.csv')
print(f'DCPP01A: {DCPP01A["label"].value_counts()}')

CINMS17B = pd.read_csv('../labeled_data/spectrograms/HARP/CINMS17B_annotations.csv')
print(f'CINMS17B: {CINMS17B["label"].value_counts()}')

CINMS18B = pd.read_csv('../labeled_data/spectrograms/HARP/CINMS18B_annotations.csv')
print(f'CINMS18B: {CINMS18B["label"].value_counts()}')

CINMS19B = pd.read_csv('../labeled_data/spectrograms/HARP/CINMS19B_annotations.csv')
print(f'CINMS19B: {CINMS19B["label"].value_counts()}')

SOCAL26H = pd.read_csv('../labeled_data/spectrograms/HARP/SOCAL26H_annotations.csv')
print(f'SOCAL26H: {SOCAL26H["label"].value_counts()}')

SOCAL34N = pd.read_csv('../labeled_data/spectrograms/HARP/SOCAL34N_annotations.csv')
print(f'SOCAL34N: {SOCAL34N["label"].value_counts()}')

# Define a color map for the categories
color_map = {
    'D': 'skyblue',
    '20Hz': 'green',
    '40Hz': 'orange',
    'A NE Pacific': 'red',
    'B NE Pacific': 'purple'
}

# Define all categories for the x-axis
categories = ['D', '20Hz', '40Hz', 'A NE Pacific', 'B NE Pacific']

# Function to create histograms with colors, fixed x-axis categories, and save as JPEG
def plot_histogram(df, title, save_path):
    label_counts = df['label'].value_counts().reindex(categories, fill_value=0)
    colors = [color_map.get(label, 'gray') for label in categories]
    
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar', color=colors)
    plt.title(f'Number of Each Call Type in {title}')
    plt.xlabel('Call Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Save the plot as a JPEG file
    file_name = f'{title}.jpeg'
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path, format='jpeg')
    
# Create histograms for each dataset
plot_histogram(CalCOFI_2004_07, 'CalCOFI_2004_07', save_path)
plot_histogram(CalCOFI_2006_02, 'CalCOFI_2006_02', save_path)
plot_histogram(CalCOFI_2006_07, 'CalCOFI_2006_07', save_path)
plot_histogram(CalCOFI_2007_07, 'CalCOFI_2007_07', save_path)
plot_histogram(CalCOFI_2007_11, 'CalCOFI_2007_11', save_path)
plot_histogram(CalCOFI_2011_04, 'CalCOFI_2011_04', save_path)
plot_histogram(CalCOFI_2011_08, 'CalCOFI_2011_08', save_path)
plot_histogram(CalCOFI_2011_11, 'CalCOFI_2011_11', save_path)
plot_histogram(CalCOFI_2015_01, 'CalCOFI_2015_01', save_path)
plot_histogram(CalCOFI_2017_01, 'CalCOFI_2017_01', save_path)
plot_histogram(CalCOFI_2017_08, 'CalCOFI_2017_08', save_path)
plot_histogram(CalCOFI_2019_07, 'CalCOFI_2019_07', save_path)

plot_histogram(DCPP01A, 'DCPP01A', save_path)
plot_histogram(CINMS17B, 'CINMS17B', save_path)
plot_histogram(CINMS18B, 'CINMS18B', save_path)
plot_histogram(CINMS19B, 'CINMS19B', save_path)
plot_histogram(SOCAL26H, 'SOCAL26H', save_path)
plot_histogram(SOCAL34N, 'SOCAL34N', save_path)
# List of all DataFrames
HARP_train_df = [
    DCPP01A, CINMS17B, CINMS18B, CINMS19B,
    SOCAL26H]

# Concatenate all DataFrames into one
HARP_train = pd.concat(HARP_train_df, ignore_index=True)

plot_histogram(HARP_train, 'HARP_train', save_path)
plot_histogram(CalCOFI_train, 'CalCOFI_train', save_path)


print(f'HARP training data:{HARP_train["label"].value_counts()}')

combined_train_df  = [CalCOFI_2004_07, CalCOFI_2006_02, CalCOFI_2006_07, CalCOFI_2007_07,
                 CalCOFI_2007_11, CalCOFI_2011_04, CalCOFI_2011_08,
                 CalCOFI_2011_11, CalCOFI_2015_01, CalCOFI_2017_01, CalCOFI_2017_08,
                 CalCOFI_2019_07,DCPP01A, CINMS17B, CINMS18B, CINMS19B,
                 SOCAL26H]

combined_train = pd.concat(combined_train_df, ignore_index= True)

print(f'Combined HARP + CalCOFI train: {combined_train["label"].value_counts()}')

test_df = [CalCOFI_2008_08, SOCAL34N]
combined_test = pd.concat(test_df, ignore_index=True)

plot_histogram(combined_train, 'Combined_train', save_path)

plot_histogram(combined_test, 'Combined_test', save_path)


print(f'Combined HARP + CalCOFI test: {combined_test["label"].value_counts()}')


# Define a function to filter out invalid bounding boxes
def filter_positive_bounding_boxes(df):
    # Check that all bounding boxes have positive width and height
    valid_boxes = (df['xmax'] > df['xmin']) & (df['ymax'] > df['ymin'])
    # Filter the DataFrame to keep only valid boxes
    return df[valid_boxes].reset_index(drop=True)

# Define a function to remove duplicate bounding boxes for the same image
def remove_duplicate_boxes(df):
    # Remove duplicates based on the spectrogram path and bounding box coordinates
    df = df.drop_duplicates(subset=['spectrogram_path', 'xmin', 'ymin', 'xmax', 'ymax']).reset_index(drop=True)
    return df

# Apply both functions to the DataFrame
def process_dataframe(df):
    df = filter_positive_bounding_boxes(df)
    df = remove_duplicate_boxes(df)
    return df

All_test = process_dataframe(combined_test)

print(f'All test: {All_test["label"].value_counts()}')

All_train = process_dataframe(combined_train)

print(f'All train: {All_train["label"].value_counts()}')


All_train.to_csv('../labeled_data/train_val_test_annotations/train.csv', index=False)
combined_test.to_csv('../labeled_data/train_val_test_annotations/test.csv', index=False)