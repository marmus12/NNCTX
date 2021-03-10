#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:21:19 2021

@author: root
"""
#### CRITERION#########
n_uctx = 2137
ucounts = np.zeros((2137,2))
for i in range(10000):
    
    if train_batch_dests[i] == 0:
        
        ucounts[ic[i],0] = ucounts[ic[i],0]+1
    else:
        ucounts[ic[i],1] = ucounts[ic[i],1]+1

CL = 0
for j in range(n_uctx):
    
    CL+=ucounts[j,0]*np.log2(network_out(utrain_ctxs[j,:]))+ucounts[j,1]*np.log2(1-network_out(utrain_ctxs[j,:]))
    ########
