#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:30:32 2021

@author: root
"""

from pcloud_functs import pcread,lowerResolution
import numpy as np
import os, sys


import tensorflow.compat.v1 as tf1
from models import tfint10_3
from ac_functs import ac_model2
from usefuls import show_time_spent,compare_Locations,get_dir_size

import globz

# print("hello")
globz.init()
from datetime import datetime
import inspect
from shutil import copyfile
from config_utils import get_model_info
#%%

#%%#CONFIGURATION
model_type='NNOC'
assert(model_type in ['NNOC','fNNOC','fNNOC1','fNNOC2','fNNOC3'])
GPU = 0
decode=1

filepath = '/path/to/plys/redandblack_vox10_1550.ply' 

nlevel_down = 0 #set number of times to downsample the input to compress a lower resolution

#%%
ckpt_dir = 'trained_models/'
output_root = 'output_root/'
if not GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


enc_functs_file,log_id,ctx_type = get_model_info(model_type)
exec('from '+enc_functs_file+ ' import ENCODE_DECODE')
ckpt_path = ckpt_dir+log_id+'/checkpoint.npy'
nn_model = tfint10_3(ckpt_path)
#################################################################


curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(output_root):
    os.mkdir(output_root)
output_dir = output_root + curr_date + '/'
os.mkdir(output_dir)
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
copyfile(curr_file,output_dir + curr_date + "__" + curr_file.split("/")[-1])     
bs_dir = output_dir + 'bss/'
os.mkdir(bs_dir)
##################################################################

sess = tf1.Session()
sess.run(tf1.global_variables_initializer())

GT = pcread(filepath).astype('int')
ori_level = np.ceil(np.log2(np.max(GT))).astype(int)
assert(str(ori_level) in filepath)
#%%#LOWER RES INPUT FOR DEBUGGING:

ori_level = ori_level-nlevel_down
for il in range(nlevel_down):
    GT = lowerResolution(GT)
    
print('input level:'+str(ori_level))
#%%###################################    

acbspath = bs_dir+'AC.dat'
ac_model = ac_model2(2,acbspath,1)
_,time_spente = ENCODE_DECODE(1,bs_dir,nn_model,ac_model,sess,ori_level,GT)
npts = GT.shape[0]

CL = get_dir_size(bs_dir)
bpv = CL/npts
# bpvs[ifile]=bpv
print('bpv: '+str(bpv))
print('filepath:'+filepath)
print('input level:'+str(ori_level))

if decode:
    ac_model = ac_model2(2,acbspath,0)
    dec_GT,time_spentd = ENCODE_DECODE(0,bs_dir,nn_model,ac_model,sess,ori_level)
    
    TP,FP,FN=compare_Locations(dec_GT,GT)
    
    print('bpv: '+str(bpv))
    print('filepath:'+filepath)
    print('input level:'+str(ori_level))



print('enc:')
show_time_spent(time_spente)

if decode:
    print('dec:')
    show_time_spent(time_spentd)





