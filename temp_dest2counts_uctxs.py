#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:07:19 2021

@author: root
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime

from scipy.io import loadmat
from usefuls import setdiff


#%% config
sample_frame = 'soldier_0690'
ctx_path = '/home/emre/Documents/DATA/TempT22_DesT22_' + sample_frame +'.mat'
output_path = '/home/emre/Documents/DATA/u_ctxs_counts_'+ sample_frame + '.npy'
#%%
ctx_dict = loadmat(ctx_path)
train_ctxs = ctx_dict['TempT22']
train_dests = ctx_dict['DesT22']


####
utrain_ctxs,trinds,trinvs,utr_counts  = np.unique(train_ctxs , axis=0,return_index=True,return_inverse=True,return_counts=True)

#%%

nutrain_ctxs = utrain_ctxs.shape[0]
train_01_counts = np.zeros([nutrain_ctxs,2])
for ictx in range(nutrain_ctxs):
    ctx_dests = train_dests[trinvs==ictx]
    train_01_counts[ictx,1] = np.sum(ctx_dests)
    train_01_counts[ictx,0] =  ctx_dests.shape[0]-train_01_counts[ictx,1]
    
np.save(output_path,{'u_ctxs':utrain_ctxs,'counts':train_01_counts})

