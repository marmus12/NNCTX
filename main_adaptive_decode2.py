#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:30:32 2021

@author: root
"""

import numpy as np
from pcloud_functs import pcread,pcfshow
from scipy.io import loadmat, savemat
import os, sys
from coding_functs import CodingCross_with_nn_probs, Coding_with_nn_and_counts, Coding_with_AC
import h5py
from enc_functs import get_temps_dests2
from ac_functs import ac_model2

import time

start = time.time()
# print("hello")



#%%#CONFIGURATION

#use gpu??
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#%%

ctx_type = 122
from models import MyModel10 as mymodel

ckpt_dir = '/home/emre/Documents/train_logs/'
log_id = '20210311-150154'
ckpt_path = ckpt_dir+log_id+'/cp.ckpt'
bspath =  'bsfile.dat'




batch_size = 10000


#%%

m=mymodel(ctx_type)
m.model.load_weights(ckpt_path)

ENC = 0

dec_model = ac_model2(2,bspath,0)
Coding_with_AC(0,0,m,dec_model,batch_size,ENC)

dec_model.end_decoding()


# CL = os.path.getsize(bspath)*8
##

# bpv = CL/npts
# print('bpv: '+str(bpv))
#bpv_ctx = CL_ctx/npts

end = time.time()
time_spent = end - start
nmins = int(time_spent//60)
nsecs = int(np.round(time_spent-nmins*60))
# print('time spent:' + str(np.round(time_spent,2)))

print('time spent: ' + str(nmins) + 'm ' + str(nsecs) + 's')
#%%
# nbadrows = 0
# for ir in range(TempTm.shape[0]):
#     if not np.prod(TempT[ir,:] == TempTm[ir,:]):
#         print(str(ir))
#         nbadrows=nbadrows+1


# nbadrows2 = 0
# for ir in range(TempTm.shape[0]):
#     if not np.prod(DesT[ir,:] == DesTm[ir,:]):
#         print(str(ir))
#         nbadrows2=nbadrows2+1
