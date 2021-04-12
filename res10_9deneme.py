#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:07:58 2021

@author: root
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:30:32 2021

@author: root
"""

import numpy as np
from pcloud_functs import pcread,pcfshow,pcshow
from scipy.io import loadmat, savemat
import os, sys
from coding_functs import CodingCross_with_nn_probs, Coding_with_AC
import h5py

# import tensorflow.compat.v1 as tf1
# from models import tfint10_2
# from ac_functs import ac_model2
from usefuls import compare_Locations,plt_imshow
import time
from glob import glob
import globz
from enc_functs_fast import N_BackForth
from pcloud_functs import lowerResolution
# print("hello")

# from datetime import datetime
# import inspect
# from shutil import copyfile
#%%#CONFIGURATION
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# if ENC:
PCC_Data_Dir ='/media/emre/Data/DATA/'
output_root = '/media/emre/Data/euvip_tests/'
# fullbody
# sample = 'redandblack'#'redandblack'#'longdress'#'loot'
# body='full'
# filepaths = glob(PCC_Data_Dir + sample +  '/' +  sample +  '/Ply/*.ply')#[::-1]

#########
# upperbodies
sample = 'phil'#
ifile = 0
body='upper'


nbits = '10'
filepaths10 = glob(PCC_Data_Dir + sample + nbits +  '/ply/*.ply')
filepath10 = filepaths10[ifile]

nbits = '9'
filepaths9 = glob(PCC_Data_Dir +  sample + nbits +  '/ply/*.ply')
filepath9 = filepaths9[ifile]

#########
ctx_type = 122 



GT10 = pcread(filepath10).astype('int')
Location10 = GT10 -np.min(GT10 ,0)+32
maxesL10 = np.max(Location10,0)+[80,0,80]
LocM10 = N_BackForth(Location10 )


GT9 = pcread(filepath9).astype('int')
Location9 = GT9 -np.min(GT9 ,0)+32
maxesL9 = np.max(Location9,0)+[80,0,80]
LocM9 = N_BackForth(Location9 )


GT8 = lowerResolution(lowerResolution(GT10))
GT7=lowerResolution(GT8)

