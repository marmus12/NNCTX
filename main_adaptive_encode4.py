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

import tensorflow.compat.v1 as tf1

from ac_functs import ac_model2
from usefuls import compare_Locations,plt_imshow
import time

import globz
start = time.time()
# print("hello")
globz.init()


#%%#CONFIGURATION
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
debug = 0
bspath =  'bsfile.dat'
ENC = 0
slow = 0
ymax = 100 #crop point cloud to a maximum y for debugging
########
# if ENC:
    # if slow:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if not slow:
    # real_encoding = 1
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    globz.batch_size = 10000
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
if slow:
    from enc_functs_slow import get_temps_dests2,N_BackForth   
else:
    from enc_functs_fast2 import get_temps_dests2,N_BackForth   
    
# else:
#     from enc_functs import get_temps_dests2,N_BackForth   
    

#########
ctx_type = 122
from models import MyModel10, intModel10, intModel10_2, tfint10_2

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



ori_weights = np.load(ckpt_dir+log_id+'/weights.npy',allow_pickle=True)[()]

w1 = ori_weights['w1']
b1 = ori_weights['b1']
w2 = ori_weights['w2']
b2 = ori_weights['b2']

M = 14
Ne = 10
# np.save('data.npy',{'w1':w1,'b1':b1,'w2':w2,'b2':b2,'T':T})
C1 = 2**Ne/np.max((np.max(w1),np.max(b1)))

C2 = 2**Ne/np.max((np.max(w2),np.max(b2)))

intw1 = np.round(w1*C1).astype('int')
intb1 = np.round(b1*C1).astype('int')

intw2 = np.round(w2*C2).astype('int')
intb2 = (2**M)*np.round(b2*C2).astype('int')

intwlist = [intw1,intb1,intw2,intb2]

# m = intModel10(ctx_type,C1,C2)
# m = intModel10_2(ctx_type,C1,C2,M)
# m.model.set_weights(intwlist)

m = tfint10_2(ctx_type,C1,C2,M,intw1,intb1,intw2,intb2)

sess = tf1.Session()

sess.run(tf1.global_variables_initializer())

    # rweights = np.round(rmult*weights)/rmult
    # rwlist.append(rweights)


#######################
# m1=MyModel10r(ctx_type)
# m1.model.load_weights(ckpt_path)

# rmult = 1000
# rwlist = []
# nw = len(m1.model.weights)
# for iw in range(nw):
    
#     weights = m1.model.weights[iw]
#     rweights = np.round(rmult*weights)/rmult
#     rwlist.append(rweights)
    
# m =MyModel10r(ctx_type)
# m.model.set_weights(rwlist)
###########################



Loc_ro = np.unique(Location[:,[1,0,2]],axis=0)
Location[:,[1,0,2]] = Loc_ro
if ENC:
    globz.Loc = Location
del Loc_ro



if not ENC:
    globz.esymbs = np.load('Desds.npy')

    #%%
ac_model = ac_model2(2,bspath,ENC)



Temps,Desds,dec_Loc= get_temps_dests2(ctx_type,ENC,nn_model = m,ac_model=ac_model,maxesL = maxesL,sess=sess)




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
#%% DEBUGGING STUFF
if debug and not ENC:
    for ip in range(dec_Loc.shape[0]):
        
        if not np.prod(dec_Loc[ip,:]==Location[ip,:]):
            
            first_bad_ip = ip
            first_bad_icpc = Location[ip,1]
            break
        


#%%
if debug and ENC:
    nT = Temps.shape[0]
    
    batch_size = 1
    Tprobs1 = np.zeros((nT,2))
    nb = np.ceil(nT/batch_size).astype('int')
    for ib in range(nb):
        bTemp = Temps[ib*batch_size:(ib+1)*batch_size,:]
        Tprobs1[ib*batch_size:(ib+1)*batch_size,:] = m.model(bTemp,training=False).numpy()  
    
    batch_size = 30
    Tprobs = np.zeros((nT,2))
    nb = np.ceil(nT/batch_size).astype('int')
    for ib in range(nb):
        bTemp = Temps[ib*batch_size:(ib+1)*batch_size,:]
        Tprobs[ib*batch_size:(ib+1)*batch_size,:] = m.model(bTemp,training=False).numpy()  
    
    
    
    
    
    np.prod(Tprobs==Tprobs1)
    
    # np.sum(np.ceil(Tprobs*100).astype(int)==np.ceil(Tprobs1*100).astype(int))
    inequ = np.round(Tprobs*10).astype(int)!=np.round(Tprobs1*10).astype(int)
    np.prod(inequ)
    np.sum(inequ)
    ieinds = np.where(inequ)
    
    Tprobs[ieinds[0][0]]
    Tprobs1[ieinds[0][0]]
    
    np.round(Tprobs[ieinds[0][0]]*10).astype(int)
    np.round(Tprobs1[ieinds[0][0]]*10).astype(int)
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
if debug:
    mult=100
    s=0.5
    r=(1/mult)*s
    th = 0.001
    
    val = 0.5346453
    err = 0.0001
    
    the_range = np.arange(0,1,0.00001)
    
    table = np.zeros([len(the_range),2])
    
    for i,val in enumerate(the_range):
        
        
        table[i,0] = val
        d=val- np.floor(mult*val)/mult
        
        dist = np.abs(d-r)
        
        
        if dist < th:
            
            output = np.round(mult*val+s)
        else:
            output = np.round(mult*val)
        
        table[i,1] = output

#%%





