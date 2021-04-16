#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:30:32 2021

@author: root
"""

import numpy as np
from pcloud_functs import pcread,pcfshow,pcshow,lowerResolution
from scipy.io import loadmat, savemat
import os, sys
from coding_functs import CodingCross_with_nn_probs, Coding_with_AC
import h5py

import tensorflow.compat.v1 as tf1
from models import tfint10_2,tfint10_3
from ac_functs import ac_model2
from usefuls import compare_Locations,plt_imshow
import time
from glob import glob
import globz
from dataset import pc_ds
# print("hello")
globz.init()
from datetime import datetime
import inspect
from shutil import copyfile
#%%#CONFIGURATION
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


PCC_Data_Dir ='/media/emre/Data/DATA/'
output_root = '/media/emre/Data/euvip_tests/'

# fullbody
sample = 'loot'#'redandblack'#'longdress'#'loot'

ds = pc_ds(sample)
filepaths = ds.filepaths[0:1]
#########



level = 9
ori_level = ds.bitdepth #BE CAREFUL!!!
ENC = 1
if not ENC:
    bs_dir = '/media/emre/Data/euvip_tests/loot10_20210415-135604/bss/'
    
# slow = 0
# ymax = 10000 #crop point cloud to a maximum y for debugging
########
if ds.body=='upper':
    assert(str(ori_level) in sample)
# if not slow:
    # real_encoding = 1
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
globz.batch_size = 10000#0000
from enc_functs_fast3 import get_temps_dests2,N_BackForth   

#########
ctx_type = 100

ckpt_dir = '/home/emre/Documents/train_logs/'
log_id = '20210409-225535'#'20210415-222905'#
#################################################################

curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")
if ENC:
    output_dir = output_root + sample + str(level) +'_' + curr_date + '/'
    os.mkdir(output_dir)
    curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
    copyfile(curr_file,output_dir + curr_date + "__" + curr_file.split("/")[-1])    
 
    bs_dir = output_dir + 'bss/'
    os.mkdir(bs_dir)
##################################################################

# if slow:
#     from enc_functs_slow import get_temps_dests2,N_BackForth   
# else:
#     from enc_functs_fast2 import get_temps_dests2,N_BackForth   
    
# ori_weights = np.load(ckpt_dir+log_id+'/weights.npy',allow_pickle=True)[()]
# w1 = ori_weights['w1']
# b1 = ori_weights['b1']
# w2 = ori_weights['w2']
# b2 = ori_weights['b2']
ckpt_path = ckpt_dir+log_id+'/checkpoint.npy'
m = tfint10_3(ckpt_path)
# m = tfint10_2(ctx_type,C1,C2,M,intw1,intb1,intw2,intb2)

sess = tf1.Session()

sess.run(tf1.global_variables_initializer())

nfiles = len(filepaths)
time_spents = np.zeros((nfiles,),'int')
CLs = np.zeros((nfiles,),'int')
bpvs = np.zeros((nfiles,),'float')
for ifile,filepath in enumerate(filepaths):
    print(str(ifile) + '/' + str(nfiles))
    start = time.time()
    #%%####
    if ds.body=='full':
        iframe = filepaths[ifile].split('vox10_')[1].split('.ply')[0]
    else:
        iframe = filepaths[ifile].split('frame')[1].split('.ply')[0]
    bspath =  bs_dir +'bs_'+iframe+'.dat'
    ac_model = ac_model2(2,bspath,ENC)
    GT = pcread(filepath).astype('int')
    
    
    for il in range(ori_level-level):
        GT = lowerResolution(GT)
    
    
    Location = GT -np.min(GT ,0)+32
    maxesL = np.max(Location,0)+[80,0,80]
    LocM = N_BackForth(Location )
    LocM_ro = np.unique(LocM[:,[1,0,2]],axis=0)
    LocM[:,[1,0,2]] = LocM_ro
    del LocM_ro
    globz.LocM = LocM
    
    Loc_ro = np.unique(Location[:,[1,0,2]],axis=0)
    Location[:,[1,0,2]] = Loc_ro
    if ENC:
        globz.Loc = Location
    del Loc_ro


    
    dec_Loc= get_temps_dests2(ctx_type,ENC,nn_model = m,ac_model=ac_model,maxesL = maxesL,sess=sess)



    if not ENC:
    
        TP,FP,FN = compare_Locations(dec_Loc,Location)
    
    if ENC:
        ac_model.end_encoding()
        # if not slow:
        #     np.save('Desds.npy',globz.Desds)   
        
    
    CL = os.path.getsize(bspath)*8
    
    CLs[ifile] = CL
    npts = GT.shape[0]
    bpv = CL/npts
    bpvs[ifile]=bpv
    print('bpv: '+str(bpv))
    #bpv_ctx = CL_ctx/npts
    
    
    end = time.time()
    time_spent = end - start
    time_spents[ifile] = int(time_spent)
    nmins = int(time_spent//60)
    nsecs = int(np.round(time_spent-nmins*60))
    # print('time spent:' + str(np.round(time_spent,2)))
    
    print('time spent: ' + str(nmins) + 'm ' + str(nsecs) + 's')

ave_bpv = np.mean(bpvs)
print('ave. bpv: '+str(ave_bpv))

if ENC:
    np.save(output_dir+'info.npy',{'CLs':CLs,'times':time_spents,'bpvs':bpvs,'fpaths':filepaths,'ave_bpv':ave_bpv})


print('min bpv:'+str(np.min(bpvs)))
print('max bpv:'+str(np.max(bpvs)))
print('ave bpv:'+str(np.mean(bpvs)))






##
#%% DEBUGGING STUFF
# if debug and not ENC:
#     for ip in range(dec_Loc.shape[0]):
        
#         if not np.prod(dec_Loc[ip,:]==Location[ip,:]):
            
#             first_bad_ip = ip
#             first_bad_icpc = Location[ip,1]
#             break
        


#%%
# if debug and ENC:
#     nT = Temps.shape[0]
    
#     batch_size = 1
#     Tprobs1 = np.zeros((nT,2))
#     nb = np.ceil(nT/batch_size).astype('int')
#     for ib in range(nb):
#         bTemp = Temps[ib*batch_size:(ib+1)*batch_size,:]
#         Tprobs1[ib*batch_size:(ib+1)*batch_size,:] = m.model(bTemp,training=False).numpy()  
    
#     batch_size = 30
#     Tprobs = np.zeros((nT,2))
#     nb = np.ceil(nT/batch_size).astype('int')
#     for ib in range(nb):
#         bTemp = Temps[ib*batch_size:(ib+1)*batch_size,:]
#         Tprobs[ib*batch_size:(ib+1)*batch_size,:] = m.model(bTemp,training=False).numpy()  
    
    
    
    
    
#     np.prod(Tprobs==Tprobs1)
    
#     # np.sum(np.ceil(Tprobs*100).astype(int)==np.ceil(Tprobs1*100).astype(int))
#     inequ = np.round(Tprobs*10).astype(int)!=np.round(Tprobs1*10).astype(int)
#     np.prod(inequ)
#     np.sum(inequ)
#     ieinds = np.where(inequ)
    
#     Tprobs[ieinds[0][0]]
#     Tprobs1[ieinds[0][0]]
    
#     np.round(Tprobs[ieinds[0][0]]*10).astype(int)
#     np.round(Tprobs1[ieinds[0][0]]*10).astype(int)
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
# if debug:
#     mult=100
#     s=0.5
#     r=(1/mult)*s
#     th = 0.001
    
#     val = 0.5346453
#     err = 0.0001
    
#     the_range = np.arange(0,1,0.00001)
    
#     table = np.zeros([len(the_range),2])
    
#     for i,val in enumerate(the_range):
        
        
#         table[i,0] = val
#         d=val- np.floor(mult*val)/mult
        
#         dist = np.abs(d-r)
        
        
#         if dist < th:
            
#             output = np.round(mult*val+s)
#         else:
#             output = np.round(mult*val)
        
#         table[i,1] = output

#%%





