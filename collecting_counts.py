# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
#import plyfile
from pcloud_functs import collect_blocks, collect_counts, ctxbits2block
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


counts,ctxs = collect_counts(GT,block_shape,max_num_ctxs,curr_loc_inds)

# for cGT in blocks:
# #cGT = allcLocs[60]
#     cpcd = o3d.geometry.PointCloud()
#     cpcd.points = o3d.utility.Vector3dVector(cGT)
#     o3d.visualization.draw_geometries([cpcd]) #, zoom=0.3412,
ctxblock = ctxbits2block(ctxs[0,:],block_shape,curr_loc_inds,0)


# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

m= MyModel2()




prediction = m.model(np.expand_dims(ctxblock,0)).numpy()

tf.nn.softmax(prediction).numpy()

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

loss_fn(y_train[:1], prediction).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])



# output = model(np.ones((1,9,9,2)))



