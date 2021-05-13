# Onset Detection Using Convolutional Neural Networks
#### Overview
... TODO

#### Böck Dataset
The audio, annotations and splits can be found here:

https://drive.google.com/file/d/0B-MfhNTg9c5eTGJ6X0FOd0xDSWs/view?usp=sharing
https://drive.google.com/file/d/0B-MfhNTg9c5eUV9DRVluNjdTOGs/view?usp=sharing
https://drive.google.com/file/d/0B-MfhNTg9c5eeDktWUpjeUF3VkU/view?usp=sharing

321 files with a total of 102 minutes of music annotated with 25,927 onsets.
#### CNN Architecture
Input shape 15x80x3
... TODO

#### Short-Time Fourier Spectogram
Following "IMPROVED MUSICAL ONSET DETECTION WITH CONVOLUTIONAL NEURAL NETWORKS" by Sebastian Böck and Jan Schlüter.
1) Three magnitude spectrograms with a hop size of 10 ms and window sizes of 23 ms, 46 ms and 93 ms.
2) Apply an 80-band Mel filter from 27.5 Hz to 16 kHz and scale magnitudes logarithmically.
3) Stack the resulting spectograms on a third (depth) axis.\
The network input for a single decision consists of the frame to classify plus a context of ±70 ms (15 frames in total), from all three spectrograms.

![stft](https://github.com/JRamos1993/cnn-onset-detection/blob/main/images/stft_spectogram_example.png "Short-Time Fourier Spectogram")

#### Wavelet Transform Scalogram
Each audio from the dataset is split into frames of ±93ms with no hopping.\
A CWT is then applied to each of those frames, and an RGB scaleogram of size 15x80 is generated and stored into `dataset_transformed` folder in .png format.

![stft](https://github.com/JRamos1993/cnn-onset-detection/blob/main/images/cwt_scalogram_example.png "Wavelet Transform Scalogram (frame)")

#### Constant-Q Transform Spectogram
Created a CQT Spectogram from the whole signal.\
Converted to RGB to get the third axis.\
Considered 150ms as the correct frame.

![stft](https://github.com/JRamos1993/cnn-onset-detection/blob/main/images/cqt_spectogram_example.png "Constant-Q Transform Spectogram")

#### Constant-Q Transform Chromagram
Created a CQT Chromagram from the whole signal.\
Converted to RGB to get the third axis.\
Considered 150ms as the correct frame.

![stft](https://github.com/JRamos1993/cnn-onset-detection/blob/main/images/cqt_chromagram_example.png "Constant-Q Transform Chromagram")

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
- Parameters: `--transform=['stft','cwt']`
4) Run the training script: `python train.py`.
- Parameters: `--epochs=150` `--folds=8`

#### TODO's
- Refactor code into functions.
- Make smaller hop-size between frames possible.
- It should be possible to define frame sizes.
- Fourier Transform spectograms as a pre-processing possilibity (similar to Bock's work).
    Using three different spectogram sizes stacked on a different axis (similar to RGB channels).
- CQT Transform?
