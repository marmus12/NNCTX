#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:30:07 2021

@author: root
"""
import tensorflow as tf

import numpy as np


import open3d as o3d


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


