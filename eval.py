from config import *
from utils import *
from json import load

import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    
    model_name = 'LeNet_1Epochs_256x256_batch32'
    plot_history('history/'+model_name+'.json')


