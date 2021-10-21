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
      
class tfModel10_test():
    
    def __init__(self,ckpt_path):
        ori_weights = np.load(ckpt_path,allow_pickle=True)[()]
        w1 = ori_weights[0]
        b1 = ori_weights[1]
        w2 = ori_weights[2]
        b2 = ori_weights[3]
      
        self.ctx_type = w1.shape[0]        
        

        # intwlist = [intw1,intb1,intw2,intb2]    
        tf1.disable_eager_execution()
        self.input = tf1.placeholder(dtype='float',shape=[None,self.ctx_type])
        
        self.w1 = tf1.Variable(w1) 
        self.b1 = tf1.Variable(b1) 
        
        self.o1 = tf1.nn.relu((tf1.matmul(self.input,self.w1)+self.b1))

    
        self.w2 = tf1.Variable(w2) 
        self.b2 = tf1.Variable(b2) 

        self.output = tf1.nn.softmax((tf1.matmul(self.o1,self.w2)+self.b2))   

      
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



