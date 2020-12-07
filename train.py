import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Flatten, ZeroPadding2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from utils import *

tf.config.experimental.list_physical_devices('GPU')

def create_cnn(input_shape, num_classes):
    """
        Creates a CNN
    """

    model = Sequential()

    # print('\tC1: Convolutional 6 kernels 5x5')
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='ReLU', input_shape=input_shape, padding='same'))
    # print('\tS2: Average Pooling 2x2 stride 2x2')
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # print('\tC3: Convolutional 16 kernels 5x5')
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='ReLU', padding='valid'))
    # print('\tS4: Average Pooling 2x2 stride 2x2')
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # print('\tC5: Convolutional 120 kernels 5x5')
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='ReLU', padding='valid'))
    model.add(Flatten())
    # print('\tF6: Fully connected, 84 units')
    model.add(Dense(84, activation='ReLU'))
    # print('\tF7: Fully connected, 10 units')
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = 'adam' #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model

# loading dataset
# X_train, X_test, y_train, y_test, labels = load_dataset('dataset')


# # creating the model
# model = create_cnn(X_train[0].shape, len(labels))

# # train
# epochs = 10
# history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data = (X_test, y_test))

# y_pred = np.argmax(model.predict(X_test), axis=1)
# y_pred = keras.utils.to_categorical(y_pred, len(labels))
# print(y_pred.shape, y_test.shape)
# print(classification_report(y_test, y_pred))


train_ds, test_ds = load_dataset('dataset_raw2')
for elem in train_ds:
    print(elem)
    break
model = create_cnn((256,256,3), 8)

# train
epochs = 10
history = model.fit(train_ds, batch_size=32, epochs=epochs, validation_data = test_ds)



