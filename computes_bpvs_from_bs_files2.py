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
#LEVELS 1,2 ARE ALREADY ACCOUNTED FOR with 64 bits
#%% 
sample='loot'#COMPLETE SET

euviptest_dirs = {10:'/media/emre/Data/euvip_tests/loot10_20210410-234245/',
                  9:'/media/emre/Data/euvip_tests/loot9_20210412-121850/',
                  8:'/media/emre/Data/euvip_tests/loot8_20210412-160809/',
                  7:'/media/emre/Data/euvip_tests/loot7_20210412-182626/'}
#%%
# sample='ricardo10' #COMPLETE SET
# body= 'upper'
# euviptest_dirs = {10:'/media/emre/Data/euvip_tests/ricardo1010_20210403-002005/',
#                   9:'/media/emre/Data/euvip_tests/ricardo109_20210403-120151/',
#                   8:'/media/emre/Data/euvip_tests/ricardo108_20210403-141238/',
#                   7:'/media/emre/Data/euvip_tests/ricardo107_20210403-144830/',
#                   6:'/media/emre/Data/euvip_tests/ricardo106_20210403-153103/',
#                   5:'/media/emre/Data/euvip_tests/ricardo105_20210403-154047/',
#                   4:'/media/emre/Data/euvip_tests/ricardo104_20210403-154851/',
#                   3:'/media/emre/Data/euvip_tests/ricardo103_20210405-151511/'}
#%%
# sample='ricardo9' #COMPLETE SET
# body= 'upper'
# euviptest_dirs = {9:'/media/emre/Data/euvip_tests/ricardo99_20210405-154444/',
#                   8:'/media/emre/Data/euvip_tests/ricardo98_20210405-184754/',
#                   7:'/media/emre/Data/euvip_tests/ricardo97_20210405-192550/',
#                   6:'/media/emre/Data/euvip_tests/ricardo96_20210405-200614/',
#                   5:'/media/emre/Data/euvip_tests/ricardo95_20210405-201508/',
#                   4:'/media/emre/Data/euvip_tests/ricardo94_20210405-202221/',
#                   3:'/media/emre/Data/euvip_tests/ricardo93_20210405-202844/'}
#%%
# sample='phil10' #COMPLETE SET
# body= 'upper'
# euviptest_dirs = {10:'/media/emre/Data/euvip_tests/phil1010_20210403-165424/',
#                   9:'/media/emre/Data/euvip_tests/phil109_20210404-130711/',
#                   8:'/media/emre/Data/euvip_tests/phil108_20210404-171317/',
#                   7:'/media/emre/Data/euvip_tests/phil107_20210404-182604/',
#                   6:'/media/emre/Data/euvip_tests/phil106_20210404-195925/',
#                   5:'/media/emre/Data/euvip_tests/phil105_20210404-203258/',
#                   4:'/media/emre/Data/euvip_tests/phil104_20210404-205150/',
#                   3:'/media/emre/Data/euvip_tests/phil103_20210404-210524/'}
#%%
# sample='phil9' #COMPLETE SET
# body= 'upper'
# euviptest_dirs = {9:'/media/emre/Data/euvip_tests/phil99_20210405-013531/',
#                   8:'/media/emre/Data/euvip_tests/phil98_20210405-124549/',
#                   7:'/media/emre/Data/euvip_tests/phil97_20210405-141300/',
#                   6:'/media/emre/Data/euvip_tests/phil96_20210405-144357/',
#                   5:'/media/emre/Data/euvip_tests/phil95_20210405-145253/',
#                   4:'/media/emre/Data/euvip_tests/phil94_20210405-145834/',
#                   3:'/media/emre/Data/euvip_tests/phil93_20210405-150256/'}
#%%
# sample='redandblack'

# euviptest_dirs = {10:'/media/emre/Data/euvip_tests/redandblack10_20210411-131805/'}



ds = pc_ds(sample)
body=ds.body

levels = list(euviptest_dirs.keys())

bss_dir = euviptest_dirs[np.min(levels)] +'bss/'
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
    # bpvs = np.zeros((nfiles,),'float')
    for iif,iframe in enumerate(iframes):
        # bspath =  bs_paths[ifile]
        bspath = euviptest_dirs[level]+'bss/bs_'+iframe+'.dat'
        # iframe = bspath.split('bs_')[1].split('.dat')[0]
        
        # for fpath in filepaths:
        if body=='full':
            filepath = ply_dir + sample + '_vox10_' + iframe + '.ply'
        # npts = pcread(filepath).shape[0]
    
    
        CLs[level,iif] = os.path.getsize(bspath)*8
        # bpvs[ifile] = CL/npts
        # print(ifile)
        
for iif,iframe in enumerate(iframes):
    if iif%50==0:
        print(str(iif)+'/'+str(nframes))
    # for fpath in filepaths:
    # if body=='full':
    #     filepath = ply_dir + sample + '_vox10_' + iframe + '.ply'  
    # elif body=='upper':
    #     filepath = ply_dir +  'frame' + iframe + '.ply'          
    npts = ds.nptss[ds.iframe2ifile(iframe)]#pcread(filepath).shape[0]
    
    bpvs[iif]=(np.sum(CLs[:,iif])+64)/npts #+64 accounts for level 1,2
    
ave_bpv = np.mean(bpvs)
print('min bpv:'+str(np.min(bpvs)))
print('max bpv:'+str(np.max(bpvs)))
print('ave bpv over '+str(len(iframes)) +' frames:'+str(ave_bpv))

np.save('/media/emre/Data/euvip_tests/'+sample+'.npy',{'CLs':CLs,'bpvs':bpvs,'fpaths':filepaths,'ave_bpv':ave_bpv,'test_dirs':euviptest_dirs})
















