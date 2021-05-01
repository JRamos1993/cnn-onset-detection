# cnn-onset-detection

#### Böck Dataset
The audio, annotations and splits can be found here:

https://drive.google.com/file/d/0B-MfhNTg9c5eTGJ6X0FOd0xDSWs/view?usp=sharing
https://drive.google.com/file/d/0B-MfhNTg9c5eUV9DRVluNjdTOGs/view?usp=sharing
https://drive.google.com/file/d/0B-MfhNTg9c5eeDktWUpjeUF3VkU/view?usp=sharing

There are more than 321 audio files and annotations.

#### CNN Architecture
Input shape 15x80x3
... TODO

#### Wavelet transform pre-processing
Each audio from the dataset is split into frames of ±93ms with no hopping.
A CWT is then applied to each of those frames, and an RGB scaleogram of size 15x80 is generated and stored into `dataset_transformed` folder.

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
    └── train.py                    # Script to train with cross-validation and evaluate the CNN model with the scaleograms

#### How to use
1) Install all the required packages by running: `pip install -r requirements`.
2) Run the pre-processing script: `python pre-process.py`.
3) Run the training script: `python train.py --epochs=10`.

#### TODO's
- Make possible to run training for less than 8 folds.
- Refactor code into functions.
- Make smaller hop-size between frames possible.
- It should be possible to define frame sizes.
- Fourier Transform spectograms as a pre-processing possilibity (similar to Bock's work).
    Using three different spectogram sizes stacked on a different axis (similar to RGB channels).
- CQT Transform?