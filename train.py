import numpy as np
from os import makedirs
from os import listdir
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
from tensorflow.keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
from os.path import basename, join
from PIL import Image
from tensorflow import keras
import tensorflow as tf 
from argparse import ArgumentParser
from keras.preprocessing.image import ImageDataGenerator
import pre_process as pp
import math

def load_split_data(split_file):
    split = np.loadtxt(split_file, dtype = str)

    train_features, train_labels = [], [] # spectograms
    validation_features, validation_labels = [], [] # spectograms

    # Load all spectograms/labels into train and validation data structures
    dataset = [join('dataset_transformed', f) for f in sorted(listdir('dataset_transformed'))]
    dataset_size = len(dataset)
    i = 1
    for sample_path in dataset:
        print(f'Loading file {i}/{dataset_size}')
        sample_file_name_with_extension = basename(sample_path)
        sample_file_name = sample_file_name_with_extension.split('.')[0]
        label = sample_file_name.split('-')[0] # First character is either 1 or 0 => onset or not onset
        sample_name_clean = sample_file_name.split('-')[1]

        image = Image.open(sample_path)
        np_image = np.array(image)

        # if i > 10000: break

        if sample_name_clean in split:
            validation_features.append(np_image)
            validation_labels.append(label)
        else:
            train_features.append(np_image)
            train_labels.append(label)

        i += 1

    # Post process
    train_features = np.array(train_features)
    validation_features = np.array(validation_features)
    train_features = train_features.astype('float32') / 255.
    validation_features = validation_features.astype('float32') / 255.

    train_labels = np.array(train_labels, dtype=int)
    validation_labels = np.array(validation_labels, dtype=int)

    # train_labels = keras.utils.to_categorical(train_labels, 2)
    # validation_labels = keras.utils.to_categorical(validation_labels, 2)

    # print(np.count_nonzero(train_labels == 0) + np.count_nonzero(validation_labels == 0))
    # print(np.count_nonzero(train_labels == 1) + np.count_nonzero(validation_labels == 1))
    # exit()

    return train_features, train_labels, validation_features, validation_labels

def load_with_keras_generator():
    datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)

    # load and iterate training dataset
    train_it = datagen.flow_from_directory('dataset_transformed/', class_mode = 'binary', batch_size = 512, subset='training', target_size=(80, 15))
    validate_it = datagen.flow_from_directory('dataset_transformed/', class_mode = 'binary', batch_size = 512, subset='validation', target_size=(80, 15))

    return train_it, validate_it

def get_model():
    # Define model
    model = Sequential()
    model.add(Conv2D(10, (7, 3), input_shape = (80,15,3), padding = 'same', activation = 'relu', data_format='channels_last'))
    model.add(MaxPooling2D(pool_size = (1, 3)))
    model.add(Conv2D(20, (3, 3), input_shape = (9, 26, 10), padding = 'valid', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (1, 3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation = 'sigmoid'))
    model.add(Dense(1, activation = 'sigmoid'))

    optimizer = SGD(lr = 0.01, momentum = 0.8, clipvalue = 5)
    # optimizer = Adam(lr = 0.01)

    model.compile(loss = 'binary_crossentropy',
        optimizer = optimizer,
        metrics = get_metrics()
    )

    # print(model.summary())

    return model

def get_callbacks():
    mca = ModelCheckpoint('models/model_{epoch:03d}.h5', monitor = 'loss', save_best_only = False)
    mcb = ModelCheckpoint('models/model_best.h5', monitor = 'loss', save_best_only = True)
    mcv = ModelCheckpoint('models/model_best_val.h5', monitor = 'val_loss', save_best_only = True)
    es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 20, verbose = True)
    tb = TensorBoard(log_dir = 'logs', write_graph = True, write_images = True)

    callbacks = [mca, mcb, mcv, es, tb]

    return callbacks

def get_metrics():
    # False negatives and false positives are samples that were incorrectly classified
    # True negatives and true positives are samples that were correctly classified
    # Accuracy is the percentage of examples correctly classified => true samples/total samples
    # Precision is the percentage of predicted positives that were correctly classified => true positives/(true positives + false positives)
    # Recall is the percentage of actual positives that were correctly classified => true positives/(true positives + false negatives)
    # AUC refers to the Area Under the Curve of a Receiver Operating Characteristic curve (ROC-AUC). This metric is equal to the probability that a classifier will rank a random positive sample higher than a random negative sample.
    # AUPRC refers to Area Under the Curve of the Precision-Recall Curve. This metric computes precision-recall pairs for different probability thresholds.

    # Accuracy is not a helpful metric for this task. We can do 95%+ accuracy on this task by predicting False all the time...

    return [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

def train_fold(fold_number, train_features, train_labels, validation_features, validation_labels, epochs):
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_number} ...')

    model = get_model()

    # Trick: Because there are not many onset samples, we want extra weight on them
    # weight_for_0 = (1 / np.count_nonzero(train_labels==0))*(len(train_labels))/2.0 
    # weight_for_1 = (1 / np.count_nonzero(train_labels==1))*(len(train_labels))/2.0
    # class_weight = {0: weight_for_0, 1: weight_for_1}
    
    history = model.fit(
        train_features,
        train_labels,
        batch_size = 256, # Large batch size to to ensure that each batch has a decent chance of containing a few positive samples.
        epochs = epochs,
        validation_data = (validation_features, validation_labels),
        callbacks = get_callbacks(),
        # class_weight = class_weight # Does not work well with SGD optimizer
    )

    return history.history

def train_fold_generator(fold_number, train_it, validate_it, epochs):
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_number} ...')

    model = get_model()

    batch_size = 512

    history = model.fit(
        train_it,
        steps_per_epoch = train_it.samples // batch_size,
        validation_data = validate_it, 
        validation_steps = validate_it.samples // batch_size,
        epochs = epochs,
        callbacks = get_callbacks(),
        shuffle = True,
        batch_size = batch_size, # Large batch size to to ensure that each batch has a decent chance of containing a few positive samples.
    )

    return history.history

def evaluate_folds(scores):
    loss, accuracy, precision, recall, tp, tn, fp, fn = [], [], [], [], [], [], [], []

    for score in scores:
        loss.append(np.mean(score['loss']))
        accuracy.append(np.mean(score['accuracy']))
        precision.append(np.mean(score['precision']))
        recall.append(np.mean(score['recall']))
        tp.append(np.mean(score['tp']))
        tn.append(np.mean(score['tn']))
        fp.append(np.mean(score['fp']))
        fn.append(np.mean(score['fn']))

    prec = np.mean(precision)
    rec = np.mean(recall)
    f_measure = 2 * prec * rec / (prec + rec)
    tp, tn, fp, fn = np.mean(tp), np.mean(tn), np.mean(fp), np.mean(fn)

    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(accuracy)} (+- {np.std(accuracy)})')
    print(f'> Loss: {np.mean(loss)}')
    print(f'> Precision: {prec}')
    print(f'> Recall: {rec}')
    print(f'> F-Measure: {f_measure}')
    print(f'> True Positives: {round(tp)} True Negatives: {round(tn)}')
    print(f'> False Positives: {round(fp)} False Negatives: {round(fn)}')
    print('------------------------------------------------------------------------')

def main():
    # See if GPU is being used
    from tensorflow.python.client import device_lib
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    print(device_lib.list_local_devices())
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)

    # Argument parsing
    parser = ArgumentParser(description = 'Onset Detection Trainer')
    parser.add_argument(
        '-e', '--epochs',
        type = int,
        default = 5,
        help = 'number of epochs to train each model for')
    parser.add_argument(
        '-f', '--folds',
        type = int,
        default = 8,
        choices = range(1,9),
        help = 'number of models to train, trained and validated with different folds of the same data')
    args = parser.parse_args()

    # Load splits information
    splits_dir = join('dataset', 'splits')
    splits_files = [join(splits_dir, f) for f in sorted(listdir(splits_dir))]

    # Some constants
    scores = []
    fold_number = 1

    for split_file in splits_files:
        if fold_number > args.folds:
            break

        # Load all images associated with the fold
        # (train_features, train_labels, validation_features, validation_labels) = load_split_data(split_file)
        # (train_it, validate_it) = load_with_keras_generator()
        # (train_features, train_labels, validation_features, validation_labels) = pp.get_ffts_dataset(split_file)
        (train_features, train_labels, validation_features, validation_labels) = pp.get_cqt_dataset(split_file)

        # Train the fold
        results = train_fold(fold_number, train_features, train_labels, validation_features, validation_labels, args.epochs)
        # results = train_fold_generator(fold_number, train_it, validate_it, args.epochs)

        # Save results to array
        scores.append(results)
            
        fold_number += 1

    # Evalute and print all results
    evaluate_folds(scores)

if __name__ == '__main__':
    main()
