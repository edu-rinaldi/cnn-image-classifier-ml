import tensorflow as tf
import sys
import keras
from keras.models import Sequential
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Flatten, ZeroPadding2D

from json import dump

from utils import *
from config import *

if __name__ == "__main__":
    tf.config.experimental.list_physical_devices('GPU')

    model_name = 'LeNet' # or 'AlexNet'    to implement 

    # loading the dataset
    training_gen, validation_gen, class_indices = load_dataset('dataset')
    num_classes = len(class_indices)

    # choosing the model
    if model_name == 'LeNet':
        model = create_LeNet(training_gen.image_shape, num_classes)
    elif model_name == 'AlexNet':
        model = create_AlexNet(training_gen.image_shape, num_classes)
    else:
        print("Error: model name does not exist")
        exit(-1)
    # training
    epochs = 35
    history = model.fit(training_gen, batch_size=BATCH_SIZE, epochs=epochs, validation_data = validation_gen)

    # built model name for saving history and the model its self
    built_model_name = f'{model_name}_{epochs}Epochs_{IMG_HEIGHT}x{IMG_WIDTH}_batch{BATCH_SIZE}'

    # Save history for the built model
    save_history(history.history, built_model_name)
        
    # saving the model
    save_model(model, built_model_name)




