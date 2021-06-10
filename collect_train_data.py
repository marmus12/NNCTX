#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:57:55 2021

@author: root
"""
from dataset import pc_ds
# from glob import glob
import globz

import random
import os
from shutil import copyfile
import inspect
import numpy as np

globz.init()
#%%
from enc_functs_fast44d import get_uctxs_counts2
root_dir = '/media/emre/Data/DATA/'
#%%
samples = ['andrew10','soldier']

nfr_per_samples = [1,1]


temp_type = 36
# min_fr_count=1
out_dir = root_dir+'n36_a1s1_'+str(temp_type)+'/'

#%%
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
copyfile(curr_file,out_dir + curr_file.split("/")[-1])    


#%%

all_counts = np.zeros((0,2),'int')
all_ctxs = np.zeros((0,temp_type),'bool')
i_fr_tot=0
nfr_tot = np.sum(nfr_per_samples)
ply_paths = []

for i_sample,sample in enumerate(samples):
    print('i_sample: ' + str(i_sample))
    ds = pc_ds(sample)
    

    i_frs = random.sample(range(ds.nframes),nfr_per_samples[i_sample])#randperm(n_all,nfr_per_samples(i_sample));
    
    for i_fr in i_frs:

        i_fr_tot=i_fr_tot+1
        print(str(i_fr_tot) + '/' + str(nfr_tot))
        
        ply_paths.append( ds.filepaths[i_fr])
        GT = ds.read_frame(i_fr)

        ctxs,counts = get_uctxs_counts2(GT,temp_type)

        
        if type(ctxs)!=int:
               
            # ctxs = boolean(ctxs);
            #%% keep only the significant ones:
            # if min_fr_count>1
            #     good_ctx_inds = sum(counts,2)>=min_fr_count;
            #     ctxs = ctxs(good_ctx_inds,:);
            #     counts = counts(good_ctx_inds,:);
            #     clear good_ctx_inds
            # end
            # %%
            
            all_ctxs = np.concatenate((all_ctxs,ctxs),axis=0)#[all_ctxs ; ctxs];
            all_counts = np.concatenate((all_counts,counts),axis=0)#[all_ctxs ; ctxs];
            nall = all_ctxs.shape[0]
            
            print('ifr/nfr:  ' + str(i_fr_tot) +'/' +str(nfr_tot))
            u_ctxs,uinvs = np.unique(all_ctxs, return_inverse=1,axis=0)

            nu_ctxs = u_ctxs.shape[0]
            u_counts = np.zeros((nu_ctxs,2),'int')

            for i_actx in range(nall):
                iuctx = uinvs[i_actx]
                u_counts[iuctx,:] = u_counts[iuctx,:]+all_counts[i_actx,:]
                if i_actx%100000==0:
                    print(str(i_actx) +'/' + str(nall ))    
            
            print('number of unique contexts in '+ ds.filepaths[i_fr] +': '+str(ctxs.shape[0]))
            print('number of unique contexts up to now: '+str(nu_ctxs))
            
            all_ctxs = np.copy(u_ctxs)
            all_counts = np.copy(u_counts)
            del u_ctxs, u_counts, uinvs
            np.save(out_dir+'data.npy',{'ctxs':all_ctxs,'counts':all_counts,'ply_paths':ply_paths})
            



# ply_paths = ply_paths(1:i_fr_tot,:);



# save([out_dir 'ctxs.mat'],'all_ctxs','-v7.3');
# save([out_dir 'counts.mat'],'all_counts','-v7.3');
# save([out_dir 'ply_paths.mat'],'ply_paths','-v7.3'); 


# n_ctxs = size(all_ctxs,1);
# fileID = fopen([out_dir 'info.txt'],'w');
# fprintf(fileID,['n_ctxs: ' num2str(n_ctxs) '\n']);
# fprintf(fileID,['sum(counts(:)): ' num2str(sum(all_counts(:))) '\n']);
# fprintf(fileID,['sum(counts(:,1)): ' num2str(sum(all_counts(:,1))) '\n']);
# fprintf(fileID,['sum(counts(:,2)): ' num2str(sum(all_counts(:,2))) '\n']);

