#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 20:02:19 2021

@author: root
"""
from glob import glob
from pcloud_functs import pcread,pcshow,lowerResolution
import os
import numpy as np

#%%
sample='loot'
body= 'full'
levels = [9,10]

#%%


PCC_Data_Dir ='/media/emre/Data/DATA/'

if body=='full':
    ply_dir = PCC_Data_Dir + sample +  '/' +  sample +  '/Ply/'
    
    
filepaths = glob(ply_dir +'*.ply')#[::-1]

        # for fpath in filepaths:
filepath = filepaths[0]

GT = pcread(filepath)
GT =GT-np.min(GT,0)

mGT = np.max(GT,0)
Locs=[]

Loc = np.copy(GT)
for ilevel in range(10):
    
    Locs.append(Loc.astype(int))
    Loc = lowerResolution(Loc)
    
Locs2=[]

shifts = [0,400,600,700,750,775,788,796,802,808]
for ilevel in range(10):

    Locs2.append(Locs[ilevel]+ np.array([shifts[ilevel],0,0]) )
    
    
allLocs = np.concatenate(Locs2,0)

# pcshow(allLocs)
pcshow(Locs[6])
    
    
    
