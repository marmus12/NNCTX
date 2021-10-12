#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:31:49 2021

@author: root
"""
import tensorflow.compat.v1 as tf1
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np


    


                        
   
  
       

     

      
 
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
    
    def __init__(self,ctx_type,do_rate=0.2):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        self.model.add(layers.Dense(2*ctx_type,activation="relu"))
        self.model.add(layers.Dropout(do_rate))
        #self.model.add(layers.Dense(50,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = "softmax"))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()               
        
        
         
class MyModel10r():
    
    def __init__(self,ctx_type):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,)))  # contexts
        self.model.add(layers.Dense(2*ctx_type,activation="relu"))
        self.model.add(layers.Lambda(lambda x: keras.backend.round(1000*x)/1000))
        #self.model.add(layers.Dense(50,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = "softmax"))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()            
        
        

class intModel10():
    
    def __init__(self,ctx_type,C1,C2):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,),dtype='int64'))  # contexts
        self.model.add(layers.Dense(2*ctx_type,activation="relu"))

        #self.model.add(layers.Dense(50,activation="relu"))
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = None))
        
        self.model.add(layers.Lambda(lambda x: x/(C1*C2)))

        self.model.add(layers.Softmax())
        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()            


class intModel10_2():
    
    def __init__(self,ctx_type,C1,C2,M):
        self.ctx_type=ctx_type
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(ctx_type,),dtype='int64'))  # contexts
        self.model.add(layers.Dense(2*ctx_type,activation=None))
        self.model.add(layers.Lambda(lambda x: x/C1))
        self.model.add(layers.ReLU())
        self.model.add(layers.Lambda(lambda x: keras.backend.round(x*(2**M))))
        
        
        
        # Finally, we add a classification layer.
        self.model.add(layers.Dense(2,activation = None))
        
        self.model.add(layers.Lambda(lambda x: x/(C2*(2**M))))

        self.model.add(layers.Softmax())
        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        self.model.summary()            

class tfint10_2():
    
    def __init__(self,ctx_type,C1,C2,M,iw1,ib1,iw2,ib2):
            
        tf1.disable_eager_execution()
        self.input = tf1.placeholder(dtype='int64',shape=[None,ctx_type])
        
        self.w1 = tf1.Variable(iw1,dtype='int64') 
        self.b1 = tf1.Variable(ib1,dtype='int64') 
        
        self.o1 = tf1.nn.relu((tf1.matmul(self.input,self.w1)+self.b1)/C1)
        self.o2 = tf1.cast(tf1.round(self.o1*(2**M)),'int64')
    
        self.w2 = tf1.Variable(iw2,dtype='int64') 
        self.b2 = tf1.Variable(ib2,dtype='int64') 

        self.output = tf1.nn.softmax((tf1.matmul(self.o2,self.w2)+self.b2)/(C2*(2**M)))



class tfModel10():
    
    def __init__(self,ctx_type):
        tf1.disable_eager_execution()
        xinitializer = keras.initializers.GlorotUniform()
        zinitializer = keras.initializers.zeros()
        self.input = tf1.placeholder(dtype='float',shape=[None,ctx_type])
        
        self.w1 = tf1.Variable(xinitializer(shape=(ctx_type,2*ctx_type))) 
        self.b1 = tf1.Variable(zinitializer(shape=(2*ctx_type,))) 
        self.o1 = tf1.nn.relu((tf1.matmul(self.input,self.w1)+self.b1))

        self.w2 = tf1.Variable(xinitializer(shape=(2*ctx_type,2))) 
        self.b2 = tf1.Variable(zinitializer(shape=(2,))) 
        self.output = tf1.nn.softmax(tf1.matmul(self.o1,self.w2)+self.b2)        
        
class tfint10_3():
    
    def __init__(self,ckpt_path,M=14,Ne=10):
        ori_weights = np.load(ckpt_path,allow_pickle=True)[()]
        w1 = ori_weights[0]
        b1 = ori_weights[1]
        w2 = ori_weights[2]
        b2 = ori_weights[3]
        self.ctx_type = w1.shape[0]        
        
        C1 = 2**Ne/np.max((np.max(w1),np.max(b1)))
        C2 = 2**Ne/np.max((np.max(w2),np.max(b2)))
        
        intw1 = np.round(w1*C1).astype('int')
        intb1 = np.round(b1*C1).astype('int')
        intw2 = np.round(w2*C2).astype('int')
        intb2 = (2**M)*np.round(b2*C2).astype('int')
        
        # intwlist = [intw1,intb1,intw2,intb2]    
        tf1.disable_eager_execution()
        self.input = tf1.placeholder(dtype='int64',shape=[None,self.ctx_type])
        
        self.w1 = tf1.Variable(intw1,dtype='int64') 
        self.b1 = tf1.Variable(intb1,dtype='int64') 
        
        self.o1 = tf1.nn.relu((tf1.matmul(self.input,self.w1)+self.b1)/C1)
        self.o2 = tf1.cast(tf1.round(self.o1*(2**M)),'int64')
    
        self.w2 = tf1.Variable(intw2,dtype='int64') 
        self.b2 = tf1.Variable(intb2,dtype='int64') 

        self.output = tf1.nn.softmax((tf1.matmul(self.o2,self.w2)+self.b2)/(C2*(2**M)))       
        
class tfModel11():
    
    def __init__(self,ctx_type):
        tf1.disable_eager_execution()
        xinitializer = keras.initializers.GlorotUniform()
        zinitializer = keras.initializers.zeros()
        self.input = tf1.placeholder(dtype='float',shape=[None,ctx_type])
        self.n_latent = ctx_type//2
        self.w1 = tf1.Variable(xinitializer(shape=(ctx_type,self.n_latent))) 
        self.b1 = tf1.Variable(zinitializer(shape=(self.n_latent,))) 
        self.o1 = tf1.nn.relu((tf1.matmul(self.input,self.w1)+self.b1))

        self.w2 = tf1.Variable(xinitializer(shape=(self.n_latent,2))) 
        self.b2 = tf1.Variable(zinitializer(shape=(2,))) 
        self.output = tf1.nn.softmax(tf1.matmul(self.o1,self.w2)+self.b2)        



