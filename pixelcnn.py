#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:50:43 2021

@author: root
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from tqdm import tqdm
from pixelcnn_models import PixelCNN


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
n_residual_blocks = 2
# The data, split between train and test sets
(x, _), (y, _) = keras.datasets.mnist.load_data()
# Concatenate all of the images together
data = np.concatenate((x, y), axis=0)
# Round all pixel values less than 33% of the max 256 value to 0
# anything above this value gets rounded up to 1 so that all values are either
# 0 or 1
data = np.where(data < (0.33 * 256), 0, 1)
data = data.astype(np.float32)


pcnn = PixelCNN(input_shape,n_residual_blocks)

adam = keras.optimizers.Adam(learning_rate=0.0005)
pcnn.model.compile(optimizer=adam, loss="binary_crossentropy")

pcnn.model.summary()
pcnn.model.fit(
    x=data, y=data, batch_size=128, epochs=50, validation_split=0.1, verbose=2
)


