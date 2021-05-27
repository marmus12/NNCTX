#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:32:37 2021

@author: root
"""
from pcloud_functs import lowerResolution,pcread,pcwrite

ori_level = 12
out_level = 10
sample='boxer'
infile = '/media/emre/Data/DATA/'+sample+'_viewdep_vox12.ply'
outfile = '/media/emre/Data/DATA/'+sample+'_viewdep_vox'+str(out_level)+'.ply'

#%%
GT = pcread(infile)



for il in range(ori_level-out_level):
    GT = lowerResolution(GT)




pcwrite(GT,outfile)