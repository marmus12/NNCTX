# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
#import plyfile
from pcloud_functs import collect_blocks, collect_counts2, ctxbits2block
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




                                  
GT = np.asarray(pcd.points,'int16')                                  


# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector((GT[(GT[:,0]<200) * (GT[:,1]<600) * (GT[:,2]<300),:]))
# o3d.visualization.draw_geometries([pcd2])


                 
block_shape = [9,9,2]    
curr_loc_inds = [4,4,1]     
max_num_ctxs = 500000
# blocks = collect_blocks(GT,block_shape,max_num_blocks)


counts,ctxs = collect_counts2(GT,block_shape,max_num_ctxs,curr_loc_inds)







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

tf.compat.v1.disable_eager_execution()
tfGT = tf.compat.v1.placeholder(tf.float32,GT.shape)
tfblock_shape = tf.compat.v1.placeholder(tf.float32,(3,))
tfmaxnum = tf.compat.v1.placeholder(tf.float32)
tfcurr_loc_inds = tf.compat.v1.placeholder(tf.float32,(3,))
y = tf.py_function(func=collect_counts2, inp=[tfGT, tfblock_shape,tfmaxnum,tfcurr_loc_inds], Tout=tf.float32)


with tf.compat.v1.Session() as sess:
  # The session executes `log_huber` eagerly. Given the feed values below,
  # it will take the first branch, so `y` evaluates to 1.0 and
  # `dy_dx` evaluates to 2.0.
  sess.run(y, feed_dict={tfGT: GT,tfblock_shape: block_shape,tfmaxnum:max_num_ctxs,tfcurr_loc_inds:curr_loc_inds})

























