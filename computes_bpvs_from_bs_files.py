#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:19:06 2021

@author: root
"""

from glob import glob
from pcloud_functs import pcread
import os
import numpy as np

#%%
sample='loot'
body= 'full'
euviptest_dir = '/media/emre/Data/euvip_tests/loot_20210401-213412/'
#%%

bss_dir = euviptest_dir +'bss/'
PCC_Data_Dir ='/media/emre/Data/DATA/'

if body=='full':
    ply_dir = PCC_Data_Dir + sample +  '/' +  sample +  '/Ply/'
    
    
filepaths = glob(ply_dir +'*.ply')#[::-1]



assert(sample in bss_dir)

bs_paths = glob(bss_dir+'*.dat')

nfiles = len(bs_paths)
bpvs = np.zeros((nfiles,),'float')
for ifile in range(nfiles):
    bspath =  bs_paths[ifile]
    iframe = bspath.split('bs_')[1].split('.dat')[0]
    
    # for fpath in filepaths:
    if body=='full':
        filepath = ply_dir + sample + '_vox10_' + iframe + '.ply'
    npts = pcread(filepath).shape[0]


    CL = os.path.getsize(bspath)*8
    bpvs[ifile] = CL/npts
    print(ifile)
    
    
ave_bpv = np.mean(bpvs)
print('min bpv:'+str(np.min(bpvs)))
print('max bpv:'+str(np.max(bpvs)))
print('ave bpv:'+str(ave_bpv))

np.save(euviptest_dir+'info.npy',{'bpvs':bpvs,'fpaths':filepaths,'ave_bpv':ave_bpv})
















