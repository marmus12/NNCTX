#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:31:03 2021

@author: root
"""
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)
    
    
# Next, we build our residual block layer.
# This is just a normal residual block, but based on the PixelConvLayer.
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])
    
    
    
class model1():    
    def __init__(self,input_shape,n_residual_blocks):
        inputs = keras.Input(shape=input_shape)
        x = PixelConvLayer(
            mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
        )(inputs)
        
        for _ in range(n_residual_blocks):
            x = ResidualBlock(filters=128)(x)
        
        for _ in range(2):
            x = PixelConvLayer(
                mask_type="B",
                filters=128,
                kernel_size=1,
                strides=1,
                activation="relu",
                padding="valid",
            )(x)
        
        out = keras.layers.Conv2D(
            filters=1, kernel_size=1, strides=1, activation="sigmoid", padding="valid"
        )(x)
        
        pixel_cnn = keras.Model(inputs, out)

        
        self.model = pixel_cnn
