# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
#import plyfile
from pcloud_functs import collect_blocks, collect_counts,ctxbits2block

import open3d as o3d
from models import MyModel2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

iframe = 1060
ply_path = '/home/emre/Documents/DATA/longdress/longdress/Ply/longdress_vox10_' + str(iframe) + '.ply'

pcd = o3d.io.read_point_cloud(ply_path )
# print(pcd)
# print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd]) #, zoom=0.3412,
                                 # front=[0.4257, -0.2125, -0.8795],
                                 # lookat=[2.6172, 2.0475, 1.532],
                                  #up=[-0.0694, -0.9768, 0.2024])
                                  
                                  
GT = np.asarray(pcd.points,'int16')                                  
                 
block_shape = [9,9,2]    
curr_loc_inds = [4,4,1]     
max_num_ctxs = 100000
# blocks = collect_blocks(GT,block_shape,max_num_blocks)


#counts,ctxs = collect_counts(GT,block_shape,max_num_ctxs,curr_loc_inds)


counts = np.load('counts.npy')
ctxs = np.load('ctxs.npy')
nctxs = ctxs.shape[0]

ctxblocks = np.zeros((nctxs,)+tuple(block_shape))
for i in range(nctxs):
    ctxblocks[i,:,:,:] = ctxbits2block(ctxs[i,:],block_shape,curr_loc_inds,0)

counts = counts+1

probs = counts/np.sum(counts,1,keepdims=True)


# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

m= MyModel2()



loss_fn = tf.keras.losses.BinaryCrossentropy()



m.model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

m.model.fit(ctxblocks, probs, epochs=100)

ictx = 59
prediction = m.model(ctxblocks[ictx :(ictx +1)]).numpy()
true_label = probs[ictx:(ictx+1)]
print('prediction:' + str(prediction) + ' true_label:' + str(true_label))




