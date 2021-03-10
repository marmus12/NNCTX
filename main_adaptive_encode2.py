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




#%%#CONFIGURATION

#use gpu??
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#%%

ctx_type = 122
from models import MyModel10 as mymodel

ckpt_dir = '/home/emre/Documents/train_logs/'
ckpt_path = ckpt_dir+'20210222-233152/cp.ckpt'

PCC_Data_Dir ='/media/emre/Data/DATA/'


# fullbody
# sample = 'loot'#'redandblack'#'longdress'#'loot'
# iframe = '1200'#'1550'       #'1300'     #'1200'
# filepath = PCC_Data_Dir + sample +  '/' +  sample +  '/Ply/' +  sample +  '_vox10_' +  iframe  + '.ply'

# upperbodies
sample = 'phil10'#
iframe = '0120'#
filepath = PCC_Data_Dir +  sample +  '/ply/frame' +  iframe  + '.ply'


batch_size = 10000

#%%####




Location = pcread(filepath)

#Location = Location[:,[2,1,0]]


npts = Location.shape[0]
#pcfshow(filepath)

#%% 
# savemat('to_get_temps_dests.mat',{'Location':Location})

#%% 

f = h5py.File('Temps_Dests.mat','r')
DesTm  = np.transpose(f['DesT'][()])
TempTm= np.transpose(f['TempT'][()])
#%% 


Location= Location-np.min(Location,0)+16

# Location= np.unique(Location,axis=0)

Temps,Desds = get_temps_dests2(Location,ctx_type)


np.sum(Temps!=TempTm)
# Temps = Temps.astype('int')
# Desds = Desds.astype('int')


#

#%%

m=mymodel(ctx_type)
m.model.load_weights(ckpt_path)




# CL,n_zero_probs npts= CodingCross_with_nn_probs(TempT,DesT,m,batch_size)
# assert(n_zero_probs==0)
#CL,CL_ctx,n_zero_probs = Coding_with_nn_and_counts(TempT22,DesT22,m)

bspath =  'bsfile.dat'
enc_model = ac_model2(2,bspath,1)
Coding_with_AC(Temps,Desds,m,enc_model,batch_size)

enc_model.end_encoding()


CL = os.path.getsize(bspath)*8
##

bpv = CL/npts
print('bpv: '+str(bpv))
#bpv_ctx = CL_ctx/npts

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
