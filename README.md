# CNN For classifying images

Homework for Machine Learning course @ Sapienza University of Rome. 
Create a model for classifying images in 8 classes:
* Angle brooms
* Chopsticks
* Oatmeal box
* Plastic tray
* Plums
* Potato chips bag
* Rice bowl
* Soft drink bottle


# Models
For this homework I used two CNN inspired by LeNet and AlexNet version. 
From testing I can say that LeNet performs better.

# How to use it (Now it's configured for using CUDA)

First thing to do is:
```bash
pip install -r requirements.txt
```

Then you can use:
* `train.py` for training the two model (I suggest to train LeNet for about 15 epochs)
* `eval.py` for plotting the training history of a certain model and for evaluating it on a validation set
* `config.py` for chaning some main settings

