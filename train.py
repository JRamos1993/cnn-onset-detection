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

def load_split_data(split_file):
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

    return X_train, Y_train, X_validation, Y_validation

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

    optimizer = SGD(lr = 0.01, momentum = 0.9, clipvalue = 5)

    model.compile(loss = 'binary_crossentropy',
        optimizer = optimizer,
        metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.TruePositives(),
        ]
    )

    # print(model.summary())

    return model

def get_callbacks():
    mca = ModelCheckpoint('models/model_{epoch:03d}.h5', monitor = 'loss', save_best_only = False)
    mcb = ModelCheckpoint('models/model_best.h5', monitor = 'loss', save_best_only = True)
    mcv = ModelCheckpoint('models/model_best_val.h5', monitor = 'val_loss', save_best_only = True)
    es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 20, verbose = False)
    tb = TensorBoard(log_dir = 'logs', write_graph = True, write_images = True)

    callbacks = [mca, mcb, mcv, es, tb]

    return callbacks

def train_fold(fold_number, X_train, Y_train, X_validation, Y_validation, epochs):
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_number} ...')

    model = get_model()

    history = model.fit(X_train, Y_train,
        steps_per_epoch = len(X_train),
        # initial_epoch = initial_epoch,
        epochs = epochs,
        validation_data = (X_validation, Y_validation),
        validation_steps = len(X_validation),
        callbacks = get_callbacks(),
    )

    return history.history

def evaluate_folds(scores):
    loss = []
    binary_accuracy = []
    precision = []
    recall = []

    tp = []
    tn = []
    fp = []
    fn = []

    a = 0
    for score in scores:
        loss.append(np.mean(score['loss']))
        binary_accuracy.append(np.mean(score['binary_accuracy']))

        if a == 0:
            precision.append(np.mean(score['precision']))
            recall.append(np.mean(score['recall']))

            tp.append(np.mean(score['true_positives']))
            tn.append(np.mean(score['true_negatives']))
            fp.append(np.mean(score['false_positives']))
            fn.append(np.mean(score['false_negatives']))
        else:
            precision.append(np.mean(score['precision_' + str(a)]))
            recall.append(np.mean(score['recall_' + str(a)]))

            tp.append(np.mean(score['true_positives_' + str(a)]))
            tn.append(np.mean(score['true_negatives_' + str(a)]))
            fp.append(np.mean(score['false_positives_' + str(a)]))
            fn.append(np.mean(score['false_negatives_' + str(a)]))
        a += 1

    prec = np.mean(precision)
    rec = np.mean(recall)
    f_measure = 2 * prec * rec / (prec + rec)

    tp = np.mean(tp)
    tn = np.mean(tn)
    fp = np.mean(fp)
    fn = np.mean(fn)

    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(binary_accuracy)} (+- {np.std(binary_accuracy)})')
    print(f'> Loss: {np.mean(loss)}')
    print(f'> Precision: {prec}')
    print(f'> Recall: {rec}')
    print(f'> F-Measure: {f_measure}')
    print(f'> True Positives: {round(tp)} True Negatives: {round(tn)}')
    print(f'> False Positives: {round(fp)} False Negatives: {round(fn)}')
    print('------------------------------------------------------------------------')

def main():
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
        (X_train, Y_train, X_validation, Y_validation) = load_split_data(split_file)

        # Train the fold
        results = train_fold(fold_number, X_train, Y_train, X_validation, Y_validation, args.epochs)

        # Save results to array
        scores.append(results)
            
        fold_number += 1

    # Evalute and print all results
    evaluate_folds(scores)

if __name__ == '__main__':
    main()
