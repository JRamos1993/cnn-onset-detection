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
from sklearn.model_selection import KFold

# Some constants
image_width = 80
image_height = 15
image_size = (image_width, image_height)
image_channels = 3
batch_size = 15

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2
)

# This is the augmentation configuration we will use for testing: only rescaling for now
test_datagen = ImageDataGenerator(rescale = 1./255)

# This is a generator that will read pictures found in subfolers of 'data/train', and generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    'dataset_transformed/train',
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'binary'
)

# This is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory('dataset_transformed/validation', target_size = image_size, batch_size = batch_size, class_mode = 'binary')

test_generator = test_datagen.flow_from_directory('dataset_transformed/test', target_size = image_size, batch_size = batch_size, class_mode = 'binary')

# input_shape = image_width,image_height,image_channels
model = Sequential()
model.add(Conv2D(10, (7, 3), input_shape = (image_width, image_height, image_channels), padding = 'same', activation = 'relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size = (1, 3)))
model.add(Conv2D(20, (3, 3), input_shape = (9, 26, 10), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (1, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

optimizer = SGD(lr = 0.01, momentum = 0.9, clipvalue = 5)

model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['binary_accuracy'])

mca = ModelCheckpoint('models/model_{epoch:03d}.h5', monitor = 'loss', save_best_only = False)
mcb = ModelCheckpoint('models/model_best.h5', monitor = 'loss', save_best_only = True)
mcv = ModelCheckpoint('models/model_best_val.h5', monitor = 'val_loss', save_best_only = True)
es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 20, verbose = True)
tb = TensorBoard(log_dir = 'logs', write_graph = True, write_images = True)
callbacks = [mca, mcb, mcv, es, tb]

history = model.fit(
    train_generator, 
    epochs = 50,
    validation_data = validation_generator,
    validation_steps = 584 // batch_size,
    steps_per_epoch = 1611 // batch_size,
    callbacks = callbacks
)

# Generate generalization metrics
score = model.evaluate(test_generator, batch_size=batch_size)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')