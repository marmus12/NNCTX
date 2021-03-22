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

from ac_functs import ac_model2
from usefuls import compare_Locations,plt_imshow
import time

import globz
start = time.time()
# print("hello")
globz.init()


#%%#CONFIGURATION


bspath =  'bsfile.dat'
ENC = 0
slow = 0
ymax = 1200 #crop point cloud to a maximum y for debugging
########
if ENC:
    if slow:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if not slow:
        # real_encoding = 1
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        globz.batch_size = 2 
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
if slow:
    from enc_functs_slow import get_temps_dests2,N_BackForth   
else:
    from enc_functs_fast2 import get_temps_dests2,N_BackForth   
    
# else:
#     from enc_functs import get_temps_dests2,N_BackForth   
    

#########
ctx_type = 122
from models import MyModel10 as mymodel

ckpt_dir = '/home/emre/Documents/train_logs/'
log_id = '20210222-233152' #'20210222-233152'#''20210311-150154' #
ckpt_path = ckpt_dir+log_id+'/cp.ckpt'

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



#%%####

GT = pcread(filepath).astype('int')
GT = GT[GT[:,1]<ymax,:]
Location = GT -np.min(GT ,0)+32
maxesL = np.max(Location,0)+[80,0,80]
LocM = N_BackForth(Location )
LocM_ro = np.unique(LocM[:,[1,0,2]],axis=0)
LocM[:,[1,0,2]] = LocM_ro
del LocM_ro
globz.LocM = LocM

m=mymodel(ctx_type)
m.model.load_weights(ckpt_path)


Loc_ro = np.unique(Location[:,[1,0,2]],axis=0)
Location[:,[1,0,2]] = Loc_ro
if ENC:
    globz.Loc = Location
del Loc_ro



if not ENC:
    globz.esymbs = np.load('Desds.npy')

    #%%
ac_model = ac_model2(2,bspath,ENC)



Temps,Desds,dec_Loc= get_temps_dests2(ctx_type,ENC,nn_model = m,ac_model=ac_model,maxesL = maxesL)




if not ENC:

    TP,FP,FN = compare_Locations(dec_Loc,Location)

if ENC:
    ac_model.end_encoding()
    # if not slow:
    #     np.save('Desds.npy',globz.Desds)   
    

CL = os.path.getsize(bspath)*8


npts = GT.shape[0]
bpv = CL/npts
print('bpv: '+str(bpv))
#bpv_ctx = CL_ctx/npts


end = time.time()
time_spent = end - start
nmins = int(time_spent//60)
nsecs = int(np.round(time_spent-nmins*60))
# print('time spent:' + str(np.round(time_spent,2)))

print('time spent: ' + str(nmins) + 'm ' + str(nsecs) + 's')

##
#%%
if not ENC:
    for ip in range(dec_Loc.shape[0]):
        
        if not np.prod(dec_Loc[ip,:]==Location[ip,:]):
            
            first_bad_ip = ip
            break
        



#%%
# if not ENC:
#     y=216
    
#     dBW = np.zeros((500,500))
    
#     # [250:320,0:200]
#     xz_inds = dec_Loc[dec_Loc[:,1]==y,:][:,[0,2]]
#     dBW[xz_inds[:,0],xz_inds[:,1]] = dBW[xz_inds[:,0],xz_inds[:,1]]+2
#     plt_imshow(dBW[240:320,0:200],(20,20))
    
#     xz_inds2 = Location[Location[:,1]==y,:][:,[0,2]]
#     # dBW = np.zeros((500,500))
#     dBW[xz_inds2[:,0],xz_inds2[:,1]] = dBW[xz_inds2[:,0],xz_inds2[:,1]] +1
#     plt_imshow(dBW[240:320,0:200],(20,20))

    # xz_inds = FP[FP[:,1]==y,:][:,[0,2]]
    # dBW = np.zeros((500,500))
    # dBW[xz_inds[:,0],xz_inds[:,1]] = 2
    # plt_imshow(dBW,(20,20))

#%% DECODER###############################
############################################



