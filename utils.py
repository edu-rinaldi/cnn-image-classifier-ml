import tensorflow as tf

import keras
from keras.models import Sequential
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Flatten, ZeroPadding2D

import numpy as np

from os.path import join
from json import load, dump

import matplotlib.pyplot as plt

from config import * 



def create_LeNet(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding='same'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model.add(Flatten())
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = 'adam' #alternative 'SGD'
    METRICS = [BinaryAccuracy(), Precision(), Recall()]
    
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=METRICS)
    return model

def save_model(model, model_name, debug=False):
    if debug: print("Saving the model")
    model.save(join('models', model_name))
    if debug:
        print("Model saved.")
        print(model)

def load_dataset(dataset_path, test_size=0.33):

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=lambda img: tf.image.resize_with_pad(img, IMG_HEIGHT, IMG_WIDTH, antialias=True),
        validation_split=test_size,
        dtype=tf.float32
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        subset='training'
    )

    validation_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False,
        subset='validation'
    )

    return train_gen, validation_gen, train_gen.class_indices

def save_history(history, model_name, debug=False):
    if debug: print("Saving the history")
    with open(f'history/{model_name}.json', 'w') as json_file:
        dump(history, json_file)
    if debug: print("History correctly saved")

def plot_history(history):
    if history is not dict:
        with open(history, 'r') as json_file:
            history = load(json_file)

    acc = np.array(history['val_binary_accuracy'])
    precision = np.array(history['val_precision'])
    recall = np.array(history['val_recall'])
    loss = np.array(history['val_loss'])
    epochs = np.arange(len(acc))

    fig, axes = plt.subplots(2,2)

    axes[0,0].plot(epochs, acc)
    axes[0,0].set(xlabel='epochs', ylabel='accuracy')
    
    axes[0,1].plot(epochs, loss)
    axes[0,1].set(xlabel='epochs', ylabel='loss')
    
    axes[1,0].plot(epochs, precision)
    axes[1,0].set(xlabel='epochs', ylabel='precision')
    
    axes[1,1].plot(epochs, recall)
    axes[1,1].set(xlabel='epochs', ylabel='recall')

    fig.set_tight_layout(True)

    plt.show()