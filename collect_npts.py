#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:52:53 2021

@author: root
"""
import numpy as np
from pcloud_functs import pcread
from glob import glob

sample ='phil10'
body='upper'

PCC_Data_Dir ='/media/emre/Data/DATA/'
sample_dir = PCC_Data_Dir + sample +  '/' 
if body=='full':
    ply_dir = sample_dir +  sample +  '/Ply/'
else:    
    ply_dir = sample_dir + '/ply/'
    
    
filepaths = glob(ply_dir +'*.ply')
nframes = len(filepaths)

nptss = np.zeros((nframes,),'int')


for iif,filepath in enumerate(filepaths):
    if iif%50==0:
        print(str(iif)+'/'+str(nframes))
       
    nptss[iif] = pcread(filepath).shape[0]
    
assert(np.prod(nptss>0))
    
np.save(sample_dir+'nptss.npy',nptss) 
    

    
    
    
    
    
    
    
    