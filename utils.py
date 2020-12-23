import tensorflow as tf

import keras
from keras.models import Sequential
from keras.metrics import BinaryAccuracy, Precision, Recall, CategoricalAccuracy
from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Flatten, ZeroPadding2D, Activation, Dropout
from keras.layers.normalization import BatchNormalization

import numpy as np

from os.path import join
from json import load, dump

import itertools

import matplotlib.pyplot as plt

from config import * 

def create_AlexNet(input_shape, num_classes, regl2 = 0.0001, lr=0.0001):

    model = Sequential()

    # C1 Convolutional Layer 
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11),\
                     strides=(2,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalization before passing it to the next layer
    model.add(BatchNormalization())

    # C2 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalization
    model.add(BatchNormalization())

    # C3 Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalization
    model.add(BatchNormalization())

    # C4 Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalization
    model.add(BatchNormalization())

    # C5 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalization
    model.add(BatchNormalization())

    # Flatten
    model.add(Flatten())

    flatten_shape = (input_shape[0]*input_shape[1]*input_shape[2],)
    
    # D1 Dense Layer
    model.add(Dense(4096, input_shape=flatten_shape, kernel_regularizer=keras.regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D2 Dense Layer
    model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalization
    model.add(BatchNormalization())

    # D3 Dense Layer
    model.add(Dense(1000, kernel_regularizer=keras.regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile
    adam = keras.optimizers.Adam(lr=lr)
    METRICS = [CategoricalAccuracy(), Precision(), Recall()]
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=METRICS)

    return model

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

    optimizer = 'adam'
    METRICS = [CategoricalAccuracy(), Precision(), Recall()]
    
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
        zoom_range=0.1,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
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

    acc = np.array(history['val_categorical_accuracy'])
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


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10,10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()