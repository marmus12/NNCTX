#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:20:25 2021

@author: root
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:30:32 2021

@author: root
"""

import numpy as np
from scipy.io import loadmat, savemat
import os, sys
import h5py

import tensorflow.compat.v1 as tf1

from ac_functs import ac_model2
from usefuls import compare_Locations,plt_imshow
import time


#%%#CONFIGURATION
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# else:
#     from enc_functs import get_temps_dests2,N_BackForth   
    

#########
ctx_type = 122
from models import MyModel10

ckpt_dir = '/home/emre/Documents/train_logs/'
log_id = '20210401-145646' #'20210222-233152' #'20210222-233152'#''20210311-150154' #
ckpt_path = ckpt_dir+log_id+'/cp.ckpt'



#%% ##DONT DELETE
m1=MyModel10(ctx_type)
m1.model.load_weights(ckpt_path)
nw = len(m1.model.weights)

w1 = m1.model.weights[0].numpy()
b1 = m1.model.weights[1].numpy()
w2 = m1.model.weights[2].numpy()
b2 = m1.model.weights[3].numpy()
np.save(ckpt_dir+log_id+'/weights.npy',{'w1':w1,'w2':w2,'b1':b1,'b2':b2})