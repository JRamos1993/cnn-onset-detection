# Onset Detection Using Convolutional Neural Networks
#### Overview
... TODO

#### Böck Dataset
The audio, annotations and splits can be found here:

https://drive.google.com/file/d/0B-MfhNTg9c5eTGJ6X0FOd0xDSWs/view?usp=sharing
https://drive.google.com/file/d/0B-MfhNTg9c5eUV9DRVluNjdTOGs/view?usp=sharing
https://drive.google.com/file/d/0B-MfhNTg9c5eeDktWUpjeUF3VkU/view?usp=sharing

There are more than 321 audio files and annotations.

#### CNN Architecture
Input shape 15x80x3
... TODO

#### Wavelet Transform Pre-processing
Each audio from the dataset is split into frames of ±93ms with no hopping.
A CWT is then applied to each of those frames, and an RGB scaleogram of size 15x80 is generated and stored into `dataset_transformed` folder in .png format.

#### Other Pre-processings...
... TODO

#### Folder structure
    .
    ├── dataset                     # Contains the Böck Dataset structure
    │   ├── annotations             # Labels
    │   ├── audio                   # Audio files in .flac format
    │   └── splits                  # Splits to use for cross-validation
    ├── dataset_transformed         # Contains the scaleograms that were converted from the audio files
    ├── logs                        # Contains logs for Tensorboard
    ├── models                      # Contains the created keras model(s)
    ├── pre-process.py              # Script to generate scaleograms from audio
    └── train.py                    # Script to train and evaluate the CNN model with the scaleograms

#### How To Use
1) Install Python version 3.8.6.
2) Install all the dependencies by running: `pip install -r requirements.txt`.
3) Run the pre-processing script: `python pre-process.py`.
4) Run the training script: `python train.py --epochs=10 --folds=8`.

#### TODO's
- Refactor code into functions.
- Make smaller hop-size between frames possible.
- It should be possible to define frame sizes.
- Fourier Transform spectograms as a pre-processing possilibity (similar to Bock's work).
    Using three different spectogram sizes stacked on a different axis (similar to RGB channels).
- CQT Transform?
