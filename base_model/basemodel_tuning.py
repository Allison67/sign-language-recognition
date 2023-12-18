import numpy as np
import pandas as pd
import json
import os
import shutil
import random
import math
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import TimeDistributed
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.models import load_model


# load
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

x_val = np.load('x_val.npy')
y_val = np.load('y_val.npy')

#x_test = np.load('x_test.npy')
#y_test = np.load('y_test.npy')

# hyperparameter tuning
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [16, 32, 64]

model_history = []

for lr in learning_rates:
    for bs in batch_sizes:
        n_labels = 100 #Number of classes
        img_channel = 3 #RGB Channels
        SEQUENCE_LENGTH =22
        CROP_SIZE=224
        num_epochs = 10

        Learning_rate = lr
        batch_size = bs

        #Implement CNN-GRU Model

        #Input Layer for CNN-GRU Model
        video = Input(shape=(SEQUENCE_LENGTH,CROP_SIZE,CROP_SIZE,img_channel))  # 64, 224, 224, 3
        #Load transfer learning model, VGG + ImageNet weights
        model = applications.MobileNet(input_shape=(CROP_SIZE,CROP_SIZE,img_channel),
                                      weights="imagenet", include_top=False)
        model.trainable = False
        # Fully Connected Dense Layer
        inputs = model.output
        x = Flatten()(inputs)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.3)(x)
        cnn_out = Dense(128, activation="relu")(x)

        #CNN Layers
        LSTM_inp = Model(inputs=model.input, outputs=cnn_out)
        #Distribute CNN output by timesteps,
        encoded_frames = TimeDistributed(LSTM_inp)(video)

        #GRU Layers
        encoded_sequence = LSTM(256)(encoded_frames)
        hidden_dropout = Dropout(0.3)(encoded_sequence)
        hidden_layer = Dense(128, activation="relu")(encoded_sequence)
        outputs = Dense(n_labels, activation="softmax")(hidden_layer)

        # Contruct CNN-GRU model
        model = Model([video], outputs)

        adam_opt = keras.optimizers.Adam(learning_rate=Learning_rate)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=adam_opt,
                      metrics=["accuracy"])

        #Train Model
        hist = model.fit(x_train, y_train, batch_size=batch_size,
                        validation_data=(x_val, y_val), shuffle=True,
                        epochs=num_epochs)
        model_history.append(hist)

        filename = f'history_lr_{lr}_bs_{bs}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(hist.history, f)

            
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots(3, 3, figsize=(20, 15))

for lr in range(len(learning_rates)):
    for bs in range(len(batch_sizes)):
        learning_rate = learning_rates[lr]
        batch_size = batch_sizes[bs]
        hist = model_history[lr * 3 + bs]

        acc_ax = loss_ax[lr][bs].twinx()
        loss_ax[lr][bs].plot(hist.history['loss'], 'y', label='train loss')
        loss_ax[lr][bs].plot(hist.history['val_loss'], 'r', label='val loss')
        acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
        acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
        loss_ax[lr][bs].set_xlabel('epoch')
        loss_ax[lr][bs].set_ylabel('loss')
        acc_ax.set_ylabel('accuray')
        loss_ax[lr][bs].legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        loss_ax[lr][bs].set_title(f'learning rate: {learning_rate}, batch size: {batch_size}')
        loss_ax[lr][bs].set_ylim(0, 5)
        acc_ax.set_ylim(0, 1)

plt.savefig('hyperparameter.png')
plt.close()
