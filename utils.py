import numpy as np
import cv2 as cv
import os.path
import pathlib
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split



def get_target_label(path_img):
    return path_img.split(".")[2].split("+")[0]

# def load_dataset(path, test_size=0.33):
#     labels = ['plastic_tray', 'rice_bowl', 'soft_drink_bottle','chopsticks','Plums','Potato_Chips_bag','Oatmeal_box','Angle_Brooms']
#     idByLabel = {el:i for i, el in enumerate(labels)}

#     X = []
#     y = []

#     files = os.listdir(path)
#     for file_img in files:
#         relpath = os.path.join(path, file_img)
#         if os.path.isdir(relpath) or file_img[0] == '.': continue
        
#         # open img
#         img = cv.imread(relpath, cv.IMREAD_COLOR)
#         # img = cv.resize(img, (128,128))
#         img = np.asarray(img, dtype=np.float)/255
#         # print(img.dtype)
#         X += [img]
#         y += [idByLabel[get_target_label(file_img)]]
#     X, y = np.array(X), np.array(y)
#     y = keras.utils.to_categorical(y, len(labels))

#     # split dataset in training and test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

#     return X_train, X_test, y_train, y_test, labels

def load_dataset(dataset_path, test_size=0.33):
    data_dir = pathlib.Path(dataset_path)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, 
        validation_split=test_size,
        
        label_mode="categorical",
        subset="training", 
        seed=123)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, 
        validation_split=test_size, 
        label_mode="categorical",
        subset="validation",
        seed=123)
    
    # class_names = train_ds.class_names
    # print(class_names)
    train_ds = train_ds.map(lambda x,y: (x/255, y))
    val_ds = val_ds.map(lambda x,y: (x/255, y))
    
    return train_ds, val_ds