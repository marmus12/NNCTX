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
from dataset import pc_ds
from shutil import copyfile
import inspect
from datetime import datetime
root_test_dir ='/media/emre/Data/euvip_tests/'
#LEVELS 1,2 ARE ALREADY ACCOUNTED FOR with 64 bits
#%% 
sample='phil9'#COMPLETE SET
euviptest_dirs = {9:root_test_dir+'phil99_20210506-143455/',
                  8:root_test_dir+'phil98_20210506-192620/',
                  7:root_test_dir+'phil97_20210506-190944/',
                  6:root_test_dir+'phil96_20210507-134732/',
                  5:root_test_dir+'phil95_20210507-135650/',
                  4:root_test_dir+'phil94_20210507-140057/',
                  3:root_test_dir+'phil93_20210507-140439/'}


last_run_level = 3


#%%
output_root = '/media/emre/Data/bpv_results/'
curr_date = datetime.now().strftime("%Y%m%d-%H%M")
output_dir = output_root + sample + '_' + curr_date + '/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    print('overwriting the already existing output_dir:'+output_dir)
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
copyfile(curr_file,output_dir + curr_date + "__" + curr_file.split("/")[-1])    


ds = pc_ds(sample)
body=ds.body

levels = list(euviptest_dirs.keys())

bss_dir = euviptest_dirs[last_run_level] +'bss/'
bs_paths = glob(bss_dir+'*.dat')

iframes =[]
for bs_path in bs_paths:   
    iframes.append(bs_path.split('bs_')[1].split('.dat')[0])

#%%
nframes = len(iframes)

PCC_Data_Dir ='/media/emre/Data/DATA/'

if body=='full':
    ply_dir = PCC_Data_Dir + sample +  '/' +  sample +  '/Ply/'
else:    
    ply_dir = PCC_Data_Dir + sample +  '/ply/'
filepaths = glob(ply_dir +'*.ply')#[::-1]
CLs = np.zeros((np.max(levels)+1,nframes),'int')
bpvs = np.zeros((nframes,),'float')

for level in levels:
    
    print('level:'+str(level))
    bss_dir = euviptest_dirs[level] +'bss/'
    assert(sample in bss_dir)
    
    bs_paths = glob(bss_dir+'*.dat')
    
    nfiles = len(bs_paths)

    for iif,iframe in enumerate(iframes):

        bspath = euviptest_dirs[level]+'bss/bs_'+iframe+'.dat'
            
        CLs[level,iif] = os.path.getsize(bspath)*8

        
for iif,iframe in enumerate(iframes):
    if iif%50==0:
        print(str(iif)+'/'+str(nframes))
   
    npts = ds.nptss[ds.iframe2ifile(iframe)]#pcread(filepath).shape[0]
    
    bpvs[iif]=(np.sum(CLs[:,iif])+64)/npts #+64 accounts for level 1,2
    
ave_bpv = np.mean(bpvs)
print('min bpv:'+str(np.min(bpvs)))
print('max bpv:'+str(np.max(bpvs)))
print('ave bpv over '+str(len(iframes)) +' frames:'+str(ave_bpv))

np.save(output_dir+sample+'.npy',{'CLs':CLs,'bpvs':bpvs,'bs_paths':bs_paths,'ave_bpv':ave_bpv,'test_dirs':euviptest_dirs})
















