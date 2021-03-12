# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os, inspect
from shutil import copyfile
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Conv2D
from scipy.io import loadmat
from usefuls import setdiff,find_common_rows,find_diff_rows,find_diff_rows_with_ind
#import plyfile
from pcloud_functs import collect_blocks, collect_counts,ctxbits2block
from train_utils import CL_criterion

import open3d as o3d
#from models import MyModel4,MyModel5,MyModel6

from dataset import ctx_dataset, ctx_dataset2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%%config
num_epochs = 30000
batch_size = 5000
optimizer = 'adam'

from models import Model10d as mymodel



ctx_type=122
# train_data_dir = ['/home/emre/Documents/DATA/andrew_david_sarah_6_122/', 
#                   '/home/emre/Documents/DATA/longdress_18_122/']
# train_data_dir = '/media/emre/Data/DATA/ads6_longdress18_122/'
val_data_dir = '/media/emre/Data/DATA/a1_sol1_122/'
train_data_dir = '/media/emre/Data/DATA/ads6_ld9_sol9_122/'
# val_data_dir = '/media/emre/Data/DATA/ricardo1_soldier1_100_minco_1/'
           #     '/home/emre/Documents/DATA/soldier_1_122/']
                

train_ds = ctx_dataset2(train_data_dir,ctx_type)

val_ds = ctx_dataset2(val_data_dir,ctx_type)



#%%
m= mymodel(ctx_type)


#%%
curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = '/home/emre/Documents/train_logs/' + curr_date + '/'


os.mkdir(logdir)

checkpoint_path = logdir +'cp.ckpt'

curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
copyfile(curr_file,logdir + curr_date + "__" + curr_file.split("/")[-1])    
 


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='val_loss',
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)




#%%


#%%
#loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
loss_fn = CL_criterion




m.model.compile(optimizer=optimizer,loss=loss_fn)



print("Fit model on training data")
history = m.model.fit(
    train_ds.ctxs,
    train_ds.counts,
    batch_size=batch_size,
    epochs=num_epochs,

    callbacks=[tensorboard_callback,cp_callback],
    validation_data=(val_ds.ctxs,val_ds.counts),
)


#%%
# i=137
# T2 = np.reshape(train_ds.ctxs[i,0:9],[3,3])
# T2b = np.zeros([5,5])
# T2b[1:4,1:4] = T2

# T1 = np.reshape(train_ds.ctxs[i,9:34],[5,5]).astype('float')


#%%
#max_num_ctxs = 100000
# blocks = collect_blocks(GT,block_shape,max_num_blocks)


#counts,ctxs = collect_counts(GT,block_shape,max_num_ctxs,curr_loc_inds)

# ctx_dir = '/home/emre/Documents/DATA/ctxs_3340_longdress_1051_9_9_2/'
# #counts = np.load('counts.npy')
# #ctxs = np.load('ctxs.npy')
# counts = loadmat(ctx_dir+'counts.mat')['counts'][1:]
# existings = np.sum(counts,1)>0
# counts = counts[existings]
# ctxs  = loadmat(ctx_dir+'ctxs.mat')['ctxs'][1:]
# ctxs = ctxs[existings,:]

# ctxblocks = np.zeros((nctxs,)+tuple(block_shape))
# for i in range(nctxs):
#     ctxblocks[i,:,:,:] = ctxbits2block(ctxs[i,:],block_shape,curr_loc_inds,0)

# counts = counts+1

# probs = counts/np.sum(counts,1,keepdims=True)


# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()


#train_test split
# tr_ratio = 0.9

# n_train = np.floor(tr_ratio*nctxs).astype('int')
# n_val = nctxs-n_train
# tr_inds  = np.random.choice(range(nctxs),n_train,replace=False)
# val_inds = setdiff(range(nctxs),tr_inds)

# train_blocks = ctxblocks[tr_inds,:,:,:]
# train_probs = probs[tr_inds,:]
# val_blocks = ctxblocks[val_inds,:,:,:]
# val_probs = probs[val_inds,:]
####



#test
# ictx = 99
# prediction = m.model(ctxblocks[ictx :(ictx +1)]).numpy()
# true_label = probs[ictx:(ictx+1)]
# print('prediction:' + str(prediction) + ' true_label:' + str(true_label))




















