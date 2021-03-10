#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:30:32 2021

@author: root
"""

import numpy as np
from pcloud_functs import pcread,pcfshow
from scipy.io import loadmat,savemat
import os
from coding_functs import CodingCross_with_nn_probs,Coding_with_nn_and_counts
import h5py



#%%#CONFIGURATION

#use gpu??
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#%%

ctx_type = 122
from models import MyModel10 as mymodel

ckpt_dir = '/home/emre/Documents/train_logs/'
ckpt_path = ckpt_dir+'20210303-140108/cp.ckpt'

PCC_Data_Dir ='/media/emre/Data/DATA/'


# fullbody
sample = 'loot'#'redandblack'#'longdress'#'loot'
iframe = '1200'#'1550'       #'1300'     #'1200'
filepath = PCC_Data_Dir + sample +  '/' +  sample +  '/Ply/' +  sample +  '_vox10_' +  iframe  + '.ply'

# upperbodies
# sample = 'phil10'#
# iframe = '0120'#
# filepath = PCC_Data_Dir +  sample +  '/ply/frame' +  iframe  + '.ply'
batch_size = 1000

#%%####


Location = pcread(filepath).astype('float32')

#Location = Location[:,[2,1,0]]


npts = Location.shape[0]
#pcfshow(filepath)
savemat('to_get_temps_dests.mat',{'Location':Location})



#%% 
#run this in shell
# command = '''/usr/local/MATLAB/R2020b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('/home/emre/Documents/MATLAB_SECTION/call_get_temps_dests.m');exit;"'''
# os.system(command)
assert(False)
#%%

m=mymodel(ctx_type)
m.model.load_weights(ckpt_path)

# TDs = loadmat('Temps_Dests.mat')
# TempT = TDs['TempT']
# DesT = TDs['DesT']

f = h5py.File('Temps_Dests.mat','r')
DesT  = np.transpose(f['DesT'][()]).astype('int')
TempT= np.transpose(f['TempT'][()]).astype('bool')



CL,n_zero_probs = CodingCross_with_nn_probs(TempT,DesT,m,batch_size)
assert(n_zero_probs==0)
#CL,CL_ctx,n_zero_probs = Coding_with_nn_and_counts(TempT22,DesT22,m)


##

bpv = CL/npts
print('bpv: '+str(bpv))
#bpv_ctx = CL_ctx/npts













