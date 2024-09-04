# WhaleMoanDetector: A fine-tuned faster-rCNN for detecting and classifying blue and fin whale moans in audio data

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)

![Fin whale](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/blue_whale.jpeg)
Blue whales off the coast of San Diego.
Photo credit: Manuel Mendieta

## Overview 

This repository contains all of the necessary code to train, test, and run inference using WhaleMoanDetector, a fine-tuned faster-rCNN model for blue and fin whale calls, on wav files. (although the wav files are not uploaded)

![spectrogram](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/all_example.JPG)
WhaleMoanDetector is trained to identify blue whale A, B, and D calls, and fin whale 20 Hz and 40 Hz calls in 60 second windows. 

If you would like to retrain the model, you need access to the required wav files. The annotations are uploaded here. 

## WhaleMoanDetector Directory Structure: 

```
WhaleMoanDetector/
├── LICENSE
├── README.md          <- The top-level README for users.
├── code               <- All code required to run detector
│   ├── AudioDetectionDataset.py     <- Dataset class.
│   ├── AudioStreamDescriptor.py     <- Parses xwav and wav headers.
│   ├── custom_collate.py            <- Collate function.
│   ├── extract_wav_header.py	     <- Function to extract xwav and wav headers.
│   ├── freq_bbox.py		     <- Converts bounding boxes to frequency (Hz).
│   ├── inference_functions.py       <- Necessary functions to make audio file chunks and spectrograms, run inference,
│   │						and save out prediction spreadsheet and images
│   ├── inference_pipeline.py	     <- Wrapper to run inference_functions and modify file paths.
│   ├── make_annotations.py          <- Wrapper to make_spectrograms.py for training/validation/testing examples
│   ├── make_spectrograms.py         <- Necessary functions to make audio file chunks, spectrograms, and labels for
│   │						training, testing, and validation.
│   ├── modify_timestamp.py  	     <- Convert Triton LoggerPro files to "modified annotation" format with 
│   │						start and end times as # of seconds into start of wav file.
│   ├── modify_timestamp_function.py <- Necessary function for modify_timestamp.py.
│   ├── plot_groundtruth.py          <- Plots groundtruth annotations in Spyder software.
│   ├── test_compute_metrics.py      <- Generates PR curve for test data.
│   ├── test_plot_predictions.py     <- Visualize predictions and groundtruth labels in Spyder.
│   ├── time_bbox.py                 <- Converts bounding boxes to time (sec into file).
│   ├── train.py                     <- Train WhaleMoanDetector.
│   ├── train_val_test_split_all.py  <- Make train/val/test splits using outputs of make_annotations.py.
│   
├── figures               <- Graphics and figures to be used in reporting
├── labeled_data
│   │	└── CalCOFI
│   │	    └── modified_annotations <- Modified start and end time .csv files 
│   │	└── HARP
│   │	    └── modified_annotations <-Modified start and end time .csv files 
│   │
│   ├── spectrograms   <- The final spectrogram images used for model training and testing (not uploaded to GitHub)
│   │	└── CalCOFI 
│   │	└── HARP
│   └── wav           <- The original wav files (not uploaded to GitHub)
│
├── models             <- Trained models (not uploaded to GitHub)
   
```
## For new Python users:

1. [Download Python 3.9](https://www.python.org/downloads/release/python-390/)

2. [Download Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)

3. [Miniconda tutorial](https://docs.anaconda.com/working-with-conda/environments/)
## WhaleMoanDetector Setup

1. Clone the Repository:

    ```bash
    git clone https://github.com/m1alksne/WhaleMoanDetector.git
    cd WhaleMoanDetector
    ```

2. Create a Virtual Environment:

    ```bash
    conda create -n whalemoandetector python=3.9
    conda activate whalemoandetector
    ```

3. Install Dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run WhaleMoanDetector on wav files:

   a. Modify File Paths in the Configuration File:
      
	Example of `config.yaml`:

      ```yaml
      wav_directory: 'L:/CHNMS_audio/CHNMS_NO_01/CHNMS_NO_01_disk01_df100'
      csv_file_path: 'M:/Mysticetes/WhaleMoanDetector_outputs/CHNMS_NO_01/CHNMS_NO_01_raw_detections.csv'
      model_path: 'L:/WhaleMoanDetector/models/WhaleMoanDetector.pth'
      ```

    b. Exectute inference_pipeline.py

    ```bash
    python code/inference_pipeline.py
    ```


