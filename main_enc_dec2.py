#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:30:32 2021

@author: root
"""

import numpy as np
from pcloud_functs import pcread,pcfshow,pcshow,lowerResolution,inds2vol,vol2inds,dilate_Loc
#from scipy.io import loadmat, savemat
import os, sys
#from coding_functs import CodingCross_with_nn_probs, Coding_with_AC
import h5py

import tensorflow.compat.v1 as tf1
from models import tfint10_2,tfint10_3
from ac_functs import ac_model2
from usefuls import show_time_spent,compare_Locations,plt_imshow,write_bits,read_bits,write_ints,read_ints,get_dir_size
import time
from glob import glob
import globz
from dataset import pc_ds
# print("hello")
globz.init()
from datetime import datetime
import inspect
from shutil import copyfile
ckpt_dir = '/home/emre/Documents/train_logs/'
#%%#CONFIGURATION
from enc_functs_fast33 import ENCODE_DECODE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sample = 'ricardo10'#'redandblack'#'longdress'#'loot'
ds = pc_ds(sample)
ori_level = ds.bitdepth
ifile=0
filepath = ds.filepaths[ifile]

nlevel_down = 0
#%%
# sample = 'Thaidancer'
# filepath = '/media/emre/Data/DATA/Thaidancer_viewdep_vox12.ply'
# ori_level=12
#%%
# sample = 'boxer'
# filepath = '/media/emre/Data/DATA/boxer_viewdep_vox12.ply'
# ori_level=10

########
#globz.batch_size = 10000#0000

#########
ctx_type = 100

log_id = '20210421-180239'#'20210409-225535'#'20210415-222905'#

ckpt_path = ckpt_dir+log_id+'/checkpoint.npy'

#################################################################
PCC_Data_Dir ='/media/emre/Data/DATA/'
output_root = '/media/emre/Data/main_enc_dec/'
# if ds.body=='upper':
#     assert(str(ori_level) in sample)
curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")
# if ENC:
output_dir = output_root + sample +'_' + curr_date + '/'
os.mkdir(output_dir)
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
copyfile(curr_file,output_dir + curr_date + "__" + curr_file.split("/")[-1])     
bs_dir = output_dir + 'bss/'
os.mkdir(bs_dir)
##################################################################
# print(bs_dir)


nn_model = tfint10_3(ckpt_path)
sess = tf1.Session()
sess.run(tf1.global_variables_initializer())

GT = pcread(filepath).astype('int')
#%%#LOWER RES INPUT FOR DEBUGGING:

ori_level = ori_level-nlevel_down
for il in range(nlevel_down):
    GT = lowerResolution(GT)
    
print('input level:'+str(ori_level))
#%%###################################    
# bs_dir = '/media/emre/Data/main_enc_dec/redandblack_20210430-184615/bss/'
_,time_spente = ENCODE_DECODE(1,bs_dir,nn_model,sess,ori_level,GT)
dec_GT,time_spentd = ENCODE_DECODE(0,bs_dir,nn_model,sess,ori_level)

TP,FP,FN=compare_Locations(dec_GT,GT)

CL = get_dir_size(bs_dir)


npts = GT.shape[0]
bpv = CL/npts
# bpvs[ifile]=bpv
print('bpv: '+str(bpv))
print('sample:'+str(sample))
print('input level:'+str(ori_level))
print('enc:')
show_time_spent(time_spente)
print('dec:')
show_time_spent(time_spentd)

# time_spent = end - start
# time_spents[ifile] = int(time_spent)
# nmins = int(time_spent//60)
# nsecs = int(np.round(time_spent-nmins*60))
# print('time spent:' + str(np.round(time_spent,2)))

# print('time spent: ' + str(nmins) + 'm ' + str(nsecs) + 's')

# ave_bpv = np.mean(bpvs)
# print('ave. bpv: '+str(ave_bpv))

# if ENC:
#     np.save(output_dir+'info.npy',{'CLs':CLs,'times':time_spents,'bpvs':bpvs,'fpaths':filepaths,'ave_bpv':ave_bpv})


# print('min bpv:'+str(np.min(bpvs)))
# print('max bpv:'+str(np.max(bpvs)))
# print('ave bpv:'+str(np.mean(bpvs)))

# print(bs_dir)




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





