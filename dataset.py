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


            
        

class pc_ds:
    
    def __init__(self,sample,root_dir = '/media/emre/Data/DATA/'):
        
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



        
        
        
