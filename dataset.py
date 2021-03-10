#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:12:54 2021

@author: root
"""
import os,sys
from scipy.io import loadmat
import numpy as np
from glob import glob
import h5py
from tensorflow import keras 

class ctx_dataset:
    
    
    def __init__(self,data_dir='/home/emre/Documents/DATA/soldier_longdress_9_38/',ctx_type=38):
        
        self.ctx_type = ctx_type
        self.data_dir = data_dir
        
        if type(self.data_dir)==str:
            self.filelist = glob(self.data_dir+'*.mat')
            self.nfiles = len(self.filelist)
            
        elif type(self.data_dir)==list:
            self.filelist =[]
            for ddir in self.data_dir:
                self.filelist = self.filelist+glob(ddir+'*.mat')
            self.nfiles = len(self.filelist)  
        
        if ctx_type<=62:
            self.ctxs,self.counts  = self.read_files('normal')
        else:
            self.ctxs,self.counts  = self.read_files('h5')
            
        self.nctxs = self.ctxs.shape[0]



    def read_files(self,formaat = 'normal'):
        
        all_ctxs = np.zeros([10**8,self.ctx_type],dtype='bool')
        all_counts = np.zeros([10**8,2],dtype='int')        
        iac = 0
        print("reading ctxs...")
        for ifi in range(self.nfiles):
            
            if formaat =='normal':
                matobj = loadmat(self.filelist[ifi])['ctxs_counts'].item()
                ctxs = matobj[0][0][0]    
                counts = matobj[1][0][0]
            elif formaat == 'h5':
                f = h5py.File(self.filelist[ifi])
                obj_ref = f['ctxs_counts']['Temps'][()][0][0]
                ctxs = np.transpose(f[obj_ref][()])
                obj_ref2 = f['ctxs_counts']['counts'][()][0][0]
                counts = np.transpose(f[obj_ref2][()])
                    
            n_ctxs = ctxs.shape[0]

            
            all_ctxs[iac:(iac+n_ctxs),:] = ctxs
            all_counts[iac:(iac+n_ctxs),:] = counts
            
            iac +=n_ctxs
            
        all_ctxs = all_ctxs[0:iac,:]
        all_counts = all_counts[0:iac,:]
        del ctxs, counts
        #u_ctxs,uinds  = np.unique(all_ctxs , axis=0,return_index=True)
        u_ctxs,uinvs = np.unique(all_ctxs, axis=0,return_inverse=True)
        
        nu_ctxs = u_ctxs.shape[0]
        u_counts = np.zeros([nu_ctxs,2],dtype='int')
        # for iuctx in range(nu_ctxs):
        #     u_counts[iuctx,:] = np.sum(all_counts[uinvs==iuctx],0)
        #     if iuctx%100==0:
        #         print(iuctx)
        for i_actx in range(iac):
            iuctx = uinvs[i_actx]
            u_counts[iuctx,:] = u_counts[iuctx,:]+all_counts[i_actx,:]
            if i_actx%1000==0:
                print(str(i_actx) + '/' + str(iac))            
        
        #u_counts = all_counts[uinds,:]
        
        # if self.ctx_type == 75:       
        #     u_ctxs = u_ctxs.reshape([-1,5,5,3])
        
        
        return u_ctxs,u_counts
            

class ctx_dataset2:
    
    
    def __init__(self,data_dir='/home/emre/Documents/DATA/ads6_longdress18_122/',ctx_type=122):
        
        
        self.ctx_type = ctx_type
        self.data_dir = data_dir
        
        f = h5py.File(self.data_dir+'ctxs.mat')        
        self.ctxs = np.transpose(f['all_ctxs'][()])
        f = h5py.File(self.data_dir+'counts.mat')        
        self.counts = np.transpose(f['all_counts'][()])
        
        assert(self.ctxs.shape[0]==self.counts.shape[0])
        assert(self.ctxs.shape[1] == ctx_type)
        self.n_ctxs = self.counts.shape[0]
        
        
        