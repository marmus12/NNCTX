#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:31:49 2021

@author: root
"""

from tensorflow import keras
from tensorflow.keras import layers

# class fc_residual_layer(layers.Layer):
#     def __init__(self, units, input_dim):
#         super(Linear, self).__init__()
#         w_init = tf.random_normal_initializer()
#         self.w = tf.Variable(
#             initial_value=w_init(shape=(input_dim, units), dtype="float32"),
#             trainable=True,
#         )
#         b_init = tf.zeros_initializer()
#         self.b = tf.Variable(
#             initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
#         )

#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b

class fc_residualBlock(keras.layers.Layer):
    def __init__(self,n_out):
        super(fc_residualBlock, self).__init__()
        self.n_out = n_out
        self.dense_1 = layers.Dense(self.n_out)
        self.bn1 =  layers.BatchNormalization()
        self.lrelu1 = layers.LeakyReLU()
        self.dense_2 = layers.Dense(self.n_out)
        self.bn2 =  layers.BatchNormalization()
        self.lrelu2= layers.LeakyReLU()
       # self.adder = layers.add()
        # self.linear_2 = Linear(32)
        # self.linear_3 = Linear(1)

    def call(self, inputs):
        # x = self.linear_1(inputs)
        # x = tf.nn.relu(x)
        # x = self.linear_2(x)
        # x = tf.nn.relu(x)
     # down-sampling is performed with a stride of 2
        shortcut = inputs
        y = self.dense_1(inputs)
        y = self.bn1(y)
        y = self.lrelu1(y)
    
        y = self.dense_2(y)
        y = self.bn2(y)
    
        y = layers.add([shortcut, y])
        y = self.lrelu2(y)       
        return y
    
    
    
def fc_residual_block(y, nb_channels):#, _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Dense(nb_channels)(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Dense(nb_channels)(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    # if _project_shortcut or _strides != (1, 1):
    #     # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
    #     # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
    #     shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
    #     shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y
        
class MyModel4():
    
    def __init__(self,ctx_type=38):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        self.model.add(layers.Dense(100,activation="relu"))
        self.model.add(layers.Dense(100,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = "softmax"))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()
                        
        
class MyModel5():
    
    def __init__(self,ctx_type=38):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        self.model.add(layers.Dense(500,activation="relu"))
        self.model.add(layers.Dense(500,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = "softmax"))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()        
    
    
class MyModel6():
    
    def __init__(self,ctx_type=38):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        self.model.add(fc_residualBlock(ctx_type))
        self.model.add(fc_residualBlock(ctx_type))           
        self.model.add(fc_residualBlock(ctx_type))   
        self.model.add(fc_residualBlock(ctx_type))   
        # self.model.add(layers.Dense(1000,activation="relu"))
        # self.model.add(layers.Dense(500,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = "softmax"))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()        
       
    
class MyModel7():
    
    def __init__(self,ctx_type=38):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        self.model.add(layers.Dense(100,activation="relu"))
        self.model.add(layers.Dense(50,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = "softmax"))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()          
   
     
class MyModel8():
    
    def __init__(self,ctx_type=38):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        self.model.add(layers.Dense(100,activation="relu"))
        #self.model.add(layers.Dense(50,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = "softmax"))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()            
        
        
class MyModel9():
    
    def __init__(self,ctx_type=38):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        self.model.add(layers.Dense(ctx_type,activation="relu"))
        #self.model.add(layers.Dense(50,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = "softmax"))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()           
 
class MyModel10():
    
    def __init__(self,ctx_type):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        self.model.add(layers.Dense(2*ctx_type,activation="relu"))

        #self.model.add(layers.Dense(50,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = "softmax"))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()            
        
        
class Model10d(): #MyModel10 with a dropout layer added
    
    def __init__(self,ctx_type):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        self.model.add(layers.Dense(2*ctx_type,activation="relu"))
        self.model.add(layers.Dropout(0.2))
        #self.model.add(layers.Dense(50,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = "softmax"))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()               
        
        
        
        
        
        
# class MyModel6():
    
#     def __init__(self,ctx_type=38):
#         self.ctx_type=ctx_type
        
#         i = keras.Input(shape=(ctx_type,))
#         d = fc_residual_block(i, 1000)
#         d = fc_residual_block(d, 500)
#         d= layers.Dense(2,activation = "softmax")(d)
        
        
        # self.model = keras.Sequential()
        # self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        # self.model.add(layers.Dense(1000,activation="relu"))
        # self.model.add(layers.Dense(500,activation="relu"))
        # # Finally, we add a classification layer.
        # self.model.add(layers.Dense(2,activation = "softmax"))

        # # Can you guess what the current output shape is at this point? Probably not.
        # # Let's just print it:
        # self.model.summary()     
    
    





# Create an instance of the model
#model = MyModel()
