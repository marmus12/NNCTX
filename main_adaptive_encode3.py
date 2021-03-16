#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:30:32 2021

@author: root
"""

import numpy as np
from pcloud_functs import pcread,pcfshow,pcshow
from scipy.io import loadmat, savemat
import os, sys
from coding_functs import CodingCross_with_nn_probs, Coding_with_AC
import h5py

from ac_functs import ac_model2
from usefuls import compare_Locations,plt_imshow
import time

import globz
start = time.time()
# print("hello")



#%%#CONFIGURATION
globz.init()


ENC = 0
slow = 1
########
if ENC:
    if not slow:
        real_encoding = 1
        batch_size = 10000
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    
if slow:
    from enc_functs_slow import get_temps_dests2,N_BackForth   
else:
    from enc_functs import get_temps_dests2,N_BackForth   
    

#########
ctx_type = 122
from models import MyModel10 as mymodel

ckpt_dir = '/home/emre/Documents/train_logs/'
log_id = '20210222-233152' #'20210222-233152'#''20210311-150154' #
ckpt_path = ckpt_dir+log_id+'/cp.ckpt'

# if ENC:
PCC_Data_Dir ='/media/emre/Data/DATA/'
# fullbody
sample = 'loot'#'redandblack'#'longdress'#'loot'
iframe = '1200'#'1550'       #'1300'     #'1200'
filepath = PCC_Data_Dir + sample +  '/' +  sample +  '/Ply/' +  sample +  '_vox10_' +  iframe  + '.ply'

# upperbodies
# sample = 'phil10'#
# iframe = '0120'#
# filepath = PCC_Data_Dir +  sample +  '/ply/frame' +  iframe  + '.ply'

bspath =  'bsfile.dat'

#%%####

GT = pcread(filepath).astype('int')
GT = GT[GT[:,1]<30,:]
Location = GT -np.min(GT ,0)+24
LocM = N_BackForth(Location )
LocM_ro = np.unique(LocM[:,[1,0,2]],axis=0)
LocM[:,[1,0,2]] = LocM_ro
del LocM_ro


m=mymodel(ctx_type)
m.model.load_weights(ckpt_path)


Loc_ro = np.unique(Location[:,[1,0,2]],axis=0)
Location[:,[1,0,2]] = Loc_ro
del Loc_ro
maxesL = np.max(Location,0)


if ENC:
    
    # dec_model = 0
    ac_model = ac_model2(2,bspath,1)
    
else:
    globz.esymbs = np.load('Desds.npy')

    # maxesL = np.max(Location,0)
    #%%
    ac_model = ac_model2(2,bspath,0)



Temps,Desds,dec_Loc= get_temps_dests2(Location,ctx_type,LocM,ENC,nn_model = m,ac_model=ac_model,maxesL = maxesL)




if not ENC:

    TP,FP,FN = compare_Locations(dec_Loc,Location)

if ENC:

    if slow:
        ac_model.end_encoding()
        CL = os.path.getsize(bspath)*8
if ENC:    
    if not slow:

        #%%# simulation
        if real_encoding:
        
            enc_model = ac_model2(2,bspath,1)
            Coding_with_AC(Temps,Desds,m,enc_model,batch_size)
            enc_model.end_encoding()
            CL = os.path.getsize(bspath)*8
            
        else:
            CL,n_zero_probs = CodingCross_with_nn_probs(Temps,Desds,m,batch_size)
            assert(n_zero_probs==0)
            
        np.save('Desds.npy',Desds)   
    
    
    
    
        npts = GT.shape[0]
        bpv = CL/npts
        print('bpv: '+str(bpv))
#bpv_ctx = CL_ctx/npts


end = time.time()
time_spent = end - start
nmins = int(time_spent//60)
nsecs = int(np.round(time_spent-nmins*60))
# print('time spent:' + str(np.round(time_spent,2)))

print('time spent: ' + str(nmins) + 'm ' + str(nsecs) + 's')



#%%
if not ENC:
    y=30
    xz_inds = dec_Loc[dec_Loc[:,1]==y,:][:,[0,2]]
    dBW = np.zeros((500,500))
    dBW[xz_inds[:,0],xz_inds[:,1]] = 2
    plt_imshow(dBW[240:350,0:200],(20,20))
    
    xz_inds2 = Location[Location[:,1]==y,:][:,[0,2]]
    # dBW = np.zeros((500,500))
    dBW[xz_inds2[:,0],xz_inds2[:,1]] = dBW[xz_inds2[:,0],xz_inds2[:,1]] +1
    plt_imshow(dBW[240:350,0:200],(20,20))






# ndsymbs = len(dsymbs)

# esymbs1 = np.array(esymbs[0:ndsymbs])

# dsymbs = np.array(dsymbs)




# if __name__=="__main__":
#     main()
#%%
# nbadrows = 0
# for ir in range(TempTm.shape[0]):
#     if not np.prod(TempT[ir,:] == TempTm[ir,:]):
#         print(str(ir))
#         nbadrows=nbadrows+1


# nbadrows2 = 0
# for ir in range(TempTm.shape[0]):
#     if not np.prod(DesT[ir,:] == DesTm[ir,:]):
#         print(str(ir))
#         nbadrows2=nbadrows2+1
