#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:04:17 2018

@author: anviol

https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/live_training/shallow_net_in_keras_LT.ipynb
"""

import numpy as np
np.random.seed(23)

## Load Dependencies
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

## Load Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape
y_train.shape

y_train[0:99]

X_test.shape
X_test[0]

y_test.shape
y_test[0]

## Preprocess Data
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')

X_train /= 255
X_test /= 255

X_test[0]

n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

y_test[0]

## Build Model Architecture

# Sequential Model
model = Sequential()

# 64 Neurons in Single Hidden Layer
# Sigmoid Activation Function
# Shape of the Input (784), Number of Inputs
model.add(Dense(64, activation='sigmoid',input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

## Configure Model

# MSE Cost Function
# Learning Rate of 0.01
# Accuracy Metric
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])

## Train Model

# Batch size is 128 images to pass through model (Network). Runs until passed through all images in training set
#   1 pass through all images in training set = 1 epoch
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, y_test))

# =============================================================================
# Train on 60000 samples, validate on 10000 samples
# Epoch 1/10
# 60000/60000 [==============================] - 1s 18us/step - loss: 0.0487 - acc: 0.7228 - val_loss: 0.0471 - val_acc: 0.7290
# Epoch 2/10
# 60000/60000 [==============================] - 1s 16us/step - loss: 0.0464 - acc: 0.7369 - val_loss: 0.0450 - val_acc: 0.7432
# Epoch 3/10
# 60000/60000 [==============================] - 1s 15us/step - loss: 0.0444 - acc: 0.7477 - val_loss: 0.0432 - val_acc: 0.7533
# Epoch 4/10
# 60000/60000 [==============================] - 1s 16us/step - loss: 0.0428 - acc: 0.7563 - val_loss: 0.0417 - val_acc: 0.7620
# Epoch 5/10
# 60000/60000 [==============================] - 1s 16us/step - loss: 0.0413 - acc: 0.7637 - val_loss: 0.0403 - val_acc: 0.7671
# Epoch 6/10
# 60000/60000 [==============================] - 1s 16us/step - loss: 0.0400 - acc: 0.7715 - val_loss: 0.0390 - val_acc: 0.7753
# Epoch 7/10
# 60000/60000 [==============================] - 1s 16us/step - loss: 0.0388 - acc: 0.7771 - val_loss: 0.0379 - val_acc: 0.7798
# Epoch 8/10
# 60000/60000 [==============================] - 1s 15us/step - loss: 0.0377 - acc: 0.7825 - val_loss: 0.0368 - val_acc: 0.7857
# Epoch 9/10
# 60000/60000 [==============================] - 1s 16us/step - loss: 0.0366 - acc: 0.7882 - val_loss: 0.0359 - val_acc: 0.7916
# Epoch 10/10
# 60000/60000 [==============================] - 1s 22us/step - loss: 0.0357 - acc: 0.7933 - val_loss: 0.0350 - val_acc: 0.7975
# Out[15]: <keras.callbacks.History at 0x138766e10>
# =============================================================================

model.summary()
# =============================================================================
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_1 (Dense)              (None, 64)                50240       # Num. of Parameters at Hidden Layer 1
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                650       
# =================================================================
# Total params: 50,890
# Trainable params: 50,890
# Non-trainable params: 0
# _________________________________________________________________
# 
# =============================================================================
