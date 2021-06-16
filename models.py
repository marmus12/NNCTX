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

# import tensorflow.compat.v1.nn.matmul as mm
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
        self.model.sutf1.matmulary()        
       
    
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
        self.model.sutf1.matmulary()          
   
     
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
        self.model.sutf1.matmulary()            
        
        
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
################################################################################

class tfModel10A():
    
    def __init__(self,ctx_type,from_scratch=True,init_ws=0):
        tf1.disable_eager_execution()

        xinitializer = keras.initializers.GlorotUniform()
        zinitializer = keras.initializers.zeros()
        self.input = tf1.placeholder(dtype='float',shape=[None,ctx_type])
        if from_scratch:
            w1init = xinitializer(shape=(ctx_type,2*ctx_type))
            b1init = zinitializer(shape=(2*ctx_type,))
            hl_trainable = True
        else:
            w1init = init_ws['w1']
            b1init = init_ws['b1']
            hl_trainable=False
        self.w1 = tf1.Variable(w1init,trainable=hl_trainable) 
        self.b1 = tf1.Variable(b1init,trainable=hl_trainable) 
        self.o1 = tf1.nn.relu((tf1.matmul(self.input,self.w1)+self.b1))

        self.w2s = []
        self.b2s = []
        # self.outputs = tf1.Variable(
        for ia in range(8):
            self.w2s.append( tf1.Variable(xinitializer(shape=(2*ctx_type,2))) )
            self.b2s.append(tf1.Variable(zinitializer(shape=(2,))) )
        outs = []
        for ia in range(8):
            outs.append(tf1.nn.softmax(tf1.matmul(self.o1,self.w2s[ia])+self.b2s[ia]))

        self.outputs = tf1.concat( outs,axis=1)        
        
            # self.outputs[:,2*ia+1] = tf1.nn.softmax(mm(self.o1,self.w2s[ia])+self.b2s[ia]))
class tfint10_3A():
    
    def __init__(self,ckpt_path,hckpt_path=0,M=14,Ne=10):
        
        ori_weights = np.load(ckpt_path,allow_pickle=True)[()]   
        if not hckpt_path:
            w1 = ori_weights[0]
            b1 = ori_weights[1]
        else:
            hidden_weights = np.load(hckpt_path,allow_pickle=True)[()]
            w1 = hidden_weights[0]
            b1 = hidden_weights[1]
            
        w2s=[]
        b2s=[]
        if hckpt_path:
            init_wi = 0
        else:
            init_wi = 2
        for ia in range(8):
            w2s.append(ori_weights[init_wi+2*ia])
            b2s.append(ori_weights[init_wi+1+2*ia])
            
        self.ctx_type = w1.shape[0]        
        
        C1 = 2**Ne/np.max((np.max(w1),np.max(b1)))
        C2s=[]
        for ia in range(8):
            C2s.append(2**Ne/np.max((np.max(w2s[ia]),np.max(b2s[ia]))))
        
        intw1 = np.round(w1*C1).astype('int')
        intb1 = np.round(b1*C1).astype('int')
        
        intw2s=[]
        intb2s=[]
        for ia in range(8):
            intw2s.append(np.round(w2s[ia]*C2s[ia]).astype('int'))
            intb2s.append( (2**M)*np.round(b2s[ia]*C2s[ia]).astype('int'))
        
        # intwlist = [intw1,intb1,intw2,intb2]    
        tf1.disable_eager_execution()
        self.input = tf1.placeholder(dtype='int64',shape=[None,self.ctx_type])

        self.w1 = tf1.Variable(intw1,dtype='int64')
        self.b1 = tf1.Variable(intb1,dtype='int64') 
        
        self.o1 = tf1.nn.relu((tf1.matmul(self.input,self.w1)+self.b1)/C1)
        self.o2 = tf1.cast(tf1.round(self.o1*(2**M)),'int64')

        self.w2s=[]
        self.b2s=[]
        for ia in range(8):

        
            self.w2s.append(tf1.Variable(intw2s[ia],dtype='int64') )
            self.b2s.append(tf1.Variable(intb2s[ia],dtype='int64') )

        outs = []
        for ia in range(8):
            outs.append(tf1.nn.softmax((tf1.matmul(self.o2,self.w2s[ia])+self.b2s[ia])/(C2s[ia]*(2**M)))) 
        self.outputs = tf1.concat( outs,axis=1)       
        # self.outputs = tf1.nn.softmax((tf1.matmul(self.o2,self.w2)+self.b2)/(C2*(2**M)))         
  
# o1 = (double(T)*iw1+ib1)/C1;

# ro1 = o1.*(o1>0);

# ro2 = round(ro1*2^M);

# o2 = ro2*iw2+ib2*2^M;

# o3 = o2/(C2*2^M);

# probs = exp(o3)/sum(exp(o3))






# class intModel10(keras.Model):

#   def __init__(self,ctx_type,C1,C2,w1,b1,w2,b2):
#     super(intModel10, self).__init__()
#     self.C1 = C1
#     self.C2 = C2
#     self.w1 = w1
#     self.b1 = b1
#     self.w2 = w2
#     self.b2 = b2
#     # self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
#     # self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    
    
    

#   def call(self, inputs):
#       o1 = layers.Multiply()([inputs,self.w1])
#       o2 = layers.Add()([o1,self.b1])
#       o3 = layers.ReLU()(o2)
      
      
#       o4 = layers.Multiply()([o3,self.w2])
#       o5 = layers.Add()([o4,self.b2])
      
#       o6 = layers.Lambda(lambda x: x/(self.C1*self.C2))(o5)
#       return o6
      
      
      
      
      
    # x = layers.multiply()(inputs)
    # return self.dense2(x)



        
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
