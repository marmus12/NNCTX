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
from pcloud_functs import pcread


class ctx_dataset2:


    def __init__(self,data_dir='/path/to/data/',ctx_type=100,from_npy=True):


        self.ctx_type = ctx_type
        self.data_dir = data_dir
        if not(from_npy):
            f = h5py.File(self.data_dir+'ctxs.mat')        
            self.ctxs = np.transpose(f['all_ctxs'][()])
            f = h5py.File(self.data_dir+'counts.mat')        
            self.counts = np.transpose(f['all_counts'][()])
        else:
            data = np.load(self.data_dir+'data.npy',allow_pickle=True)
            self.ctxs = data[()]['ctxs']
            self.counts = data[()]['counts']

        assert(self.ctxs.shape[0]==self.counts.shape[0])
        assert(self.ctxs.shape[1] == ctx_type)
        self.n_ctxs = self.counts.shape[0]            
        

class pc_ds:
    
    def __init__(self,sample,root_dir = '/path/to/plys/'):
        
        self.sample = sample
        self.root_dir = root_dir 
        self.sample_dir = root_dir + sample +'/'
        
        if sample in ['loot','redandblack','longdress','soldier']:
            self.body = 'full'
            self.ply_dir = self.sample_dir +  sample +  '/Ply/'
            self.filepaths = glob(self.ply_dir+'*.ply')
        elif sample in ['thaidancer','boxer']:
            self.body = 'full'
            self.ply_dir = self.root_dir
            self.filepaths = glob(self.ply_dir + '*'+sample[1:] +'*.ply')
        else:
            self.body = 'upper'
            self.ply_dir = self.sample_dir + '/ply/'
            self.filepaths = glob(self.ply_dir+'*.ply')
        
        self.nframes = len(self.filepaths)
        
        if '10' in sample:
            self.bitdepth= 10
        elif sample in ['loot','redandblack','longdress','soldier']:
            self.bitdepth = 10
        elif '9' in sample:
            self.bitdepth = 9
        elif (len(self.filepaths)==1) and ('12' in self.filepaths[0]):
            self.bitdepth = 12
            

        try:    
            self.nptss = np.load(self.sample_dir + 'nptss.npy')
        except:
            if sample in ['thaidancer','boxer']:
                
                self.nptss = pcread(self.filepaths[0]).shape[0]
            else:
                    
                self.nptss = np.zeros((self.nframes,),'int')
                print('collecting nptss..')
                for iif,filepath in enumerate(self.filepaths):
                    if iif%50==0:
                        print(str(iif)+'/'+str(self.nframes))
           
                    self.nptss[iif] = pcread(filepath).shape[0]
        
                assert(np.prod(self.nptss>0))
        
                np.save(self.sample_dir+'nptss.npy',self.nptss) 
    
    def read_frame(self,ifile=None,iframe=None):
        
        if ifile!=None and iframe!=None:
            raise ValueError("only one of iframe or ifile should be fed")
        
        
        if ifile is not(None):
            fpath = self.filepaths[ifile]
            print('file read:'+fpath)
            return pcread(fpath) 
            
        elif iframe is not(None):
            if type(iframe)!=str:
                iframe = str(iframe)
                
            if self.body=='full':
                fpath = self.ply_dir + self.sample + '_vox'+str(self.bitdepth)+'_'+iframe+'.ply'
            else:
                fpath = self.ply_dir + 'frame'+iframe+'.ply'
            
            print('file read:'+fpath)
            return pcread(fpath) 
                
        else:
            raise ValueError("either iframe or ifile should be fed")
        
    def ifile2iframe(self,ifile):
        if self.sample in ['thaidancer','boxer']:
            return '0'
        elif self.body=='full':
            return self.filepaths[ifile].split('vox10_')[1].split('.ply')[0]
        else:
            return self.filepaths[ifile].split('frame')[1].split('.ply')[0]  
        
    def iframe2ifile(self,iframe):
        if self.sample in ['thaidancer','boxer']:
            return 0
        else:
            for ifile,filepath in enumerate(self.filepaths):
                if iframe in filepath:
                    return ifile



        
        
        
