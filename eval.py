from config import *
from utils import *

from sklearn.metrics import confusion_matrix

import keras

if __name__ == "__main__":
    
    # model_name = 'AlexNet_50Epochs_118x224_batch32'
    model_name = 'LeNet_50Epochs_256x256_batch32_da'
    plot_history('history/'+model_name+'.json')

    # loading the dataset
    training_gen, validation_gen, class_indices = load_dataset('dataset')
    num_classes = len(class_indices)

    model = keras.models.load_model('models/'+model_name)
    model.save_weights("model.h5")
    
    # keras eval.
    # model.evaluate(validation_gen)

    # scikit eval.
    target_names = []
    for key in validation_gen.class_indices:
        target_names.append(key)

    #Confution Matrix 
    Y_pred = model.predict(validation_gen)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(validation_gen.classes, y_pred)
    plot_confusion_matrix(cm, target_names, title='Confusion Matrix')
