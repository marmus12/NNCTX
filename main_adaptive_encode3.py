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
from coding_functs import CodingCross_with_nn_probs, Coding_with_nn_and_counts, Coding_with_AC
import h5py
from enc_functs import get_temps_dests2,N_BackForth
from ac_functs import ac_model2

import time

start = time.time()
# print("hello")



#%%#CONFIGURATION

#use gpu??
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#%%
ENC = 0
########
if ENC:
    real_encoding = 1
    batch_size = 10000
#########
ctx_type = 122
from models import MyModel10 as mymodel

ckpt_dir = '/home/emre/Documents/train_logs/'
log_id = '20210222-233152' #'20210222-233152'#''20210311-150154' #
ckpt_path = ckpt_dir+log_id+'/cp.ckpt'


# ckpt_path = ckpt_dir+'20210225-223023/cp.ckpt'


# if ENC:
PCC_Data_Dir ='/media/emre/Data/DATA/'
# fullbody
sample = 'loot'#'redandblack'#'longdress'#'loot'
iframe = '1200'#'1550'       #'1300'     #'1200'
filepath = PCC_Data_Dir + sample +  '/' +  sample +  '/Ply/' +  sample +  '_vox10_' +  iframe  + '.ply'

# upperbodies
# sample = 'phil10'#
# iframe = '0120'#
# filepath = PCC_Data_Dir +  sample +  '/ply/frame' +  iframe  + '.ply'


bspath =  'bsfile.dat'

#%%####



if ENC:
    GT = pcread(filepath)
    
    npts = GT.shape[0]
    
    Location= GT-np.min(GT,0)+24
    
    Temps,Desds = get_temps_dests2(Location,ctx_type)



#%%

m=mymodel(ctx_type)
m.model.load_weights(ckpt_path)



if ENC:
    #%%# simulation
    if real_encoding:
    
        enc_model = ac_model2(2,bspath,1)
        Coding_with_AC(Temps,Desds,m,enc_model,batch_size)
        enc_model.end_encoding()
        CL = os.path.getsize(bspath)*8
        
    else:
        CL,n_zero_probs = CodingCross_with_nn_probs(Temps,Desds,m,batch_size)
        assert(n_zero_probs==0)
else:
    
    #%% BB007M is encoded and decoded by GPCC
    #this part is to be erased later on
    GT = pcread(filepath)
    Location = GT -np.min(GT ,0)+24
    LocM = N_BackForth(Location )
    maxesL = np.max(Location,0)
    # del Location
    #%%
    dec_model = ac_model2(2,bspath,0)
    
    dec_Loc = get_temps_dests2(0,ctx_type,0,LocM,nn_model = m,ac_model=dec_model,maxesL = maxesL)





#%%
if(ENC):
    bpv = CL/npts
    print('bpv: '+str(bpv))
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
