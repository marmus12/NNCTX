#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:29:04 2021

@author: root
"""
from matplotlib import pyplot as plt
import numpy as np
from dataset import ctx_dataset2


train_data_dir = '/media/emre/Data/DATA/F4_ads6_ls9_100/'
# val_data_dir = '/media/emre/Data/DATA/ricardo1_soldier1_100_minco_1/'
           #     '/home/emre/Documents/DATA/soldier_1_122/']
ctx_type = 100
ds = ctx_dataset2(train_data_dir,ctx_type)


ratio_th = 0.2
ratios= np.min(ds.counts,1)/np.max(ds.counts,1)
grinds = np.where(ratios<ratio_th)[0]

