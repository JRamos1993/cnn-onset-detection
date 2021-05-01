import numpy as np
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os.path import basename, exists, join, splitext
from PIL import Image
from tensorflow import keras

# Some constants
batch_size = 15
scores = []

# Load splits information
splits_dir = join('dataset', 'splits')
splits_files = [join(splits_dir, f) for f in sorted(listdir(splits_dir))]

fold_no = 1
for split_file in splits_files:
    split = np.loadtxt(split_file, dtype=str)

    X_train = [] # spectograms
    Y_train = [] # labels

    X_validation = [] # spectograms
    Y_validation = [] # labels

    # Load all spectograms/labels into train and validation data structures
    dataset = [join('dataset_transformed', f) for f in sorted(listdir('dataset_transformed'))]
    for sample_path in dataset:
        sample_file_name_with_extension = basename(sample_path)
        sample_file_name = sample_file_name_with_extension.split('.')[0]
        label = sample_file_name.split('-')[0] # First character is either 1 or 0 => onset or not onset
        sample_name_clean = sample_file_name.split('-')[1]

        image = Image.open(sample_path)
        np_image = np.array(image)

        if sample_name_clean in split:
            X_validation.append(np_image)
            Y_validation.append(label)
        else:
            X_train.append(np_image)
            Y_train.append(label)

    # Post process
    X_train = np.array(X_train)
    X_validation = np.array(X_validation)
    X_train = X_train.astype('float32') / 255.
    X_validation = X_validation.astype('float32') / 255.

    Y_train = np.array(Y_train, dtype=int)
    Y_validation = np.array(Y_validation, dtype=int)
    # Y_train = keras.utils.to_categorical(Y_train, 2)
    # Y_validation = keras.utils.to_categorical(Y_validation, 2)

    # Define model
    model = Sequential()
    model.add(Conv2D(10, (7, 3), input_shape = (15,80,3), padding = 'same', activation = 'relu', data_format='channels_last'))
    model.add(MaxPooling2D(pool_size = (1, 3)))
    model.add(Conv2D(20, (3, 3), input_shape = (9, 26, 10), padding = 'valid', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (1, 3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation = 'sigmoid'))
    model.add(Dense(1, activation = 'sigmoid'))

    optimizer = SGD(lr = 0.01, momentum = 0.9, clipvalue = 5)

    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['binary_accuracy'])

    # mca = ModelCheckpoint('models/model_{epoch:03d}.h5', monitor = 'loss', save_best_only = False)
    # mcb = ModelCheckpoint('models/model_best.h5', monitor = 'loss', save_best_only = True)
    # mcv = ModelCheckpoint('models/model_best_val.h5', monitor = 'val_loss', save_best_only = True)
    # es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 20, verbose = False)
    # tb = TensorBoard(log_dir = 'logs', write_graph = True, write_images = True)
    # callbacks = [mca, mcb, mcv, es, tb]

    # print(model.summary())

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    history = model.fit(X_train, Y_train,
        batch_size = batch_size,
        epochs = 10,
        verbose = 1,
        # callbacks = callbacks,
        validation_data=(X_validation, Y_validation),
    )

    # Evaluate the model with the validation subset
    results = history.history # model.evaluate(X_validation, Y_validation, verbose = 0)
    scores.append(results)

    # Increase fold number
    fold_no = fold_no + 1

loss = []
binary_accuracy = []
# val_loss = []
# val_binary_accuracy = []
i = 0

# # == Provide average scores ==
# print('------------------------------------------------------------------------')
# print('Score per fold')
for score in scores:
    c_loss = np.mean(score['loss'])
    c_binary_accuracy = np.mean(score['binary_accuracy'])
    # c_val_loss = np.mean(score['val_loss'])
    # c_val_binary_accuracy = np.mean(score['val_binary_accuracy'])

    loss.append(c_loss)
    binary_accuracy.append(c_binary_accuracy)
    # val_loss.append(c_val_loss)
    # val_binary_accuracy.append(c_val_binary_accuracy)

    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {c_loss} - Accuracy: {c_binary_accuracy}%')
    
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(binary_accuracy)} (+- {np.std(binary_accuracy)})')
print(f'> Loss: {np.mean(loss)}')
print('------------------------------------------------------------------------')