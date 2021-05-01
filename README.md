# cnn-onset-detection

#### Böck Dataset
The audio, annotations and splits can be found here:

https://drive.google.com/file/d/0B-MfhNTg9c5eTGJ6X0FOd0xDSWs/view?usp=sharing
https://drive.google.com/file/d/0B-MfhNTg9c5eUV9DRVluNjdTOGs/view?usp=sharing
https://drive.google.com/file/d/0B-MfhNTg9c5eeDktWUpjeUF3VkU/view?usp=sharing

There are more than 321 audio files and annotations.

#### Folder structure
    .
    ├── dataset                     # Contains the Böck Dataset structure
    │   ├── annotations             # Labels
    │   ├── audio                   # Audio files in .flac format
    │   └── splits                  # Splits to use for cross-validation
    ├── dataset_transformed         # Contains the scaleograms that were converted from the audio files
    ├── logs                        # Contains logs for Tensorboard
    ├── models                      # Contains the created kera model(s)
    └── ...

#### How to use
1) Install requirements by running: `pip install -r requirements`
2) Pre-process ...

#### TODO's
- Refactor code into functions.
- Make smaller hop-size between frames possible.
- It should be possible to define frame sizes.
- Fourier Transform spectograms as a pre-processing possilibity (similar to Bock's work).
    Using three different spectogram sizes stacked on a different axis (similar to RGB channels).
- CQT Transform?