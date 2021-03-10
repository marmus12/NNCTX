# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Conv2D
from scipy.io import loadmat
from usefuls import setdiff
#import plyfile
from pcloud_functs import collect_blocks, collect_counts,ctxbits2block






import open3d as o3d
from models import MyModel2,MyModel3



logdir = "/home/emre/Documents/train_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = logdir +"cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#iframe = 1060
#ply_path = '/home/emre/Documents/DATA/longdress/longdress/Ply/longdress_vox10_' + str(iframe) + '.ply'

#pcd = o3d.io.read_point_cloud(ply_path )
#GT = np.asarray(pcd.points,'int16')   
# print(pcd)
# print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd]) #, zoom=0.3412,

                                  
                                  
#GT = np.asarray(pcd.points,'int16')                                  
                 
# block_shape = [9,9,2]    
# curr_loc_inds = [4,4,1]     


ctx_path = '/home/emre/Documents/DATA/TempT22_DesT22_longdress_1051.mat'
val_ctx_path = '/home/emre/Documents/DATA/TempT22_DesT22_soldier_0690.mat'

ctx_dict = loadmat(ctx_path)
train_ctxs = ctx_dict['TempT22']
train_dests = ctx_dict['DesT22']



val_ctx_dict = loadmat(val_ctx_path)
val_ctxs = val_ctx_dict['TempT22']
val_dests = val_ctx_dict['DesT22']
####
utrain_ctxs,trinds,trinvs,utr_counts  = np.unique(train_ctxs , axis=0,return_index=True,return_inverse=True,return_counts=True)

uval_ctxs,valinds,valinvs,uval_counts  = np.unique(val_ctxs , axis=0,return_index=True,return_inverse=True,return_counts=True)


nutrain_ctxs = utrain_ctxs.shape[0]
train_01_counts = np.zeros([nutrain_ctxs,2])
for ictx in range(nutrain_ctxs):
    ctx_dests = train_dests[trinvs==ictx]
    train_01_counts[ictx,1] = np.sum(ctx_dests)
    train_01_counts[ictx,0] = ctx_dests.shape[0]-train_01_counts[ictx,1]
    
utrain_probs = train_01_counts/np.sum(train_01_counts,1,keepdims=True)

train_probs = utrain_probs[trinvs,:]
########
nuval_ctxs = uval_ctxs.shape[0]
val_01_counts = np.zeros([nuval_ctxs,2])
for ictx in range(nuval_ctxs):
    ctx_dests = val_dests[valinvs==ictx]
    val_01_counts[ictx,1] = np.sum(ctx_dests)
    val_01_counts[ictx,0] = ctx_dests.shape[0]-val_01_counts[ictx,1]
    
uval_probs = val_01_counts/np.sum(val_01_counts,1,keepdims=True)

val_probs = uval_probs[valinvs,:]

####

# tf_train_labels = tf.one_hot(train_labels,2).numpy()[:,0,:]
# tf_val_labels = tf.one_hot(val_labels,2).numpy()[:,0,:]

n_train = train_ctxs.shape[0]
n_val = val_ctxs.shape[0]
###
loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)


m= MyModel3()
m.model.compile(optimizer='rmsprop',
              loss=loss_fn,
              metrics=['accuracy'])

# m.model.fit(ctxblocks, probs, epochs=100)

print("Fit model on training data")
history = m.model.fit(
    train_ctxs,
    train_probs,
    batch_size=10000,
    epochs=10000,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    callbacks=[tensorboard_callback,cp_callback],
    validation_data=(val_ctxs,val_probs),
)




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




















