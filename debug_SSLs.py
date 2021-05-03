#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:31:05 2021

@author: root
"""

import numpy as np
from usefuls import read_bits,plt_imshow,read_ints
from matplotlib import pyplot as plt
from pcloud_functs import lowerResolution

bs_dir = '/media/emre/Data/main_enc_dec/loot_20210503-164948/bss/'


ori_level=10
nintbs = nintbits = ori_level*np.ones((6,),int)
minmaxesG = read_ints(nintbs,bs_dir+'maxes_mins.dat')

minmaxesG =read_ints(nintbits,bs_dir+'maxes_mins.dat')
lrmm = np.copy(minmaxesG[np.newaxis,:])
lrmms=np.zeros((ori_level+1,6),int)
lrmms[ori_level] = lrmm
for il in range(ori_level-2):
    lrmm = lowerResolution(lrmm)
    lrmms[ori_level-il-1,:] = lrmm


# SSLs  =dict()
# lens = np.zeros((11,),int)
# for level in range(3,11):
#     SSL = read_bits(bs_dir+'SSL'+str(level)+'.dat')[(level+2):-1]
#     lens[level] = len(SSL)
#     SSLs[level] = SSL
SSLs  = np.zeros((ori_level+1,2000),int)
for level in range(3,ori_level+1):
    SSL = read_bits(bs_dir+'SSL'+str(level)+'.dat')[(level+2):-1]
    for ib,bit in enumerate(SSL):
        SSLs[level,ib] = int(bit)
        
# plt_imshow(SSLs[:,30:200],(30,30))
# add =1

for level in range(4,ori_level+1):
    # lrmms[level][1]%2
    add = lrmms[level][1]%2#1-np.where(SSLs[level])[0][-1]%2
    assert(np.prod( np.where(SSLs[level-1])[0] == (lowerResolution(np.where(SSLs[level])[0]+add-32)+32) ))
    print(level)
    
    
##############################################################
dSSLs  = np.zeros((ori_level+1,2000),int)
dSSL = read_bits(bs_dir+'SSL'+str(ori_level)+'.dat')[(ori_level+2):-1]
for ib,bit in enumerate(dSSL):
    dSSLs[ori_level,ib] = int(bit) 
    
    
for level in range(ori_level,3,-1):
    # lrmms[level][1]%2
    add = lrmms[level][1]%2#1-np.where(SSLs[level])[0][-1]%2
    inds = lowerResolution(np.where(dSSLs[level])[0]+add-32)+32
    dSSLs[level-1,inds] = 1
    print(level)    
    

assert ( np.prod(dSSLs==SSLs)  )
    
    
    
    
    
    
    
    
    
    
    