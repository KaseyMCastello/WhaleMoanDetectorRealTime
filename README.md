# WhaleMoanDetector: A fine-tuned faster-rCNN for detecting blue and fin whale moans in audio data

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)

![Fin whale](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/fin_whale.JPG)
Fin whale off the coast of San Diego.

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
│   ├── AudioDetectionDataset.py     
│   ├── AudioStreamDescriptor.py
│   ├── extract_wav_header.py
│   ├── freq_bbox.py
│   ├── inference_functions.py
│   ├── inference_pipeline.py
│   ├── make_annotations.py
│   ├── make_spectrograms.py
│   ├── modify_timestamp.py
│   ├── modify_timestamp_function.py
│   ├── plot_groundtruth.py
│   ├── test_compute_metrics.py
│   ├── test_plot_predictions.py
│   ├── time_bbox.py
│   ├── train.py
│
├── figures               <- Graphics and figures to be used in reporting
├── labeled_data
│   ├── logs           <- Original annotation files
│   │	└── CalCOFI
│   │	    └── modified_annotations <- Modified start and end time .csv files 
│   │	└── HARP
│   │	    └── modified_annotations <-Modified start and end time .csv files 
│   │
│   ├── spectrograms   <- The final spectrogram images used for model training and testing (not uploaded)
│   │	└── CalCOFI 
│   │	└── HARP
│   └── wav           <- The original wav files (not uploaded to GitHub)
│
├── models             <- Trained models
   
```

## Setup

1. Clone the Repository:

	```git clone  https://github.com/m1alksne/WhaleMoanDetector.git```

	```cd WhaleMoanDetector```

2. Create a Virtual Environment:

	```conda create -n whalemoandetector python=3.9```

	```conda activate whalemoandetector```

3. Install Dependencies

        ```pip install -r requirements.txt``` 

4. Run WhaleMoanDetector on wav files

 	 ```python code/intference_pipeline.py

