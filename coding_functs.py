#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:58:47 2021

@author: root
"""
import numpy as np
import sys
sys.path.append('/home/emre/Documents/kodlar/Reference-arithmetic-coding-master/python/')

import arithmeticcoding as arc

def Coding_with_AC(Temps,Desds,model,ac_model,batch_size):
    # if ENC:
    nTemps = Temps.shape[0]
    n_batches = np.ceil(nTemps/batch_size).astype('int')
    for i_batch in range(n_batches):
        
        symbs = Desds[i_batch*batch_size:(i_batch+1)*batch_size]
        b_Temps = Temps[i_batch*batch_size:(i_batch+1)*batch_size,:]
        b_probs = model.model(b_Temps).numpy()        
        b_freqs = np.round(b_probs*1000000).astype('int')
        for icont in range(b_probs.shape[0]):
            symb = symbs[icont]   
            freqlist = list(b_freqs[icont,:])
            freqs = arc.CheckedFrequencyTable(arc.SimpleFrequencyTable(freqlist )) #THIS HAS TO BE CHECKEDFREQTABLE
            ac_model.encode_symbol(freqs,symb)
            
        if i_batch%10==0:
            print(str(i_batch) + '/' + str(n_batches) )
    # else: ##DECODER
    #     ctx_type = model.model.get_input_shape_at(0)[1]
    #     wsize = np.sqrt((ctx_type+0.5)*2/5).astype(int)
    #     b = (wsize-1)//2        
    #     [ir,ic] = np.where(np.ones((wsize,wsize))) 
    #     Tempaxa= [ic-b,ir-b]
    #     T12size = wsize**2    
    #     TCsize = (T12size-1)//2
    #     ###
    #     Temps = np.zeros((nTemps,ctx_type))
    #     Temps[0,:] = np.zeros((ctx_type,),'int')
    #     for iT in range(nTemps):
    #         Temp = Temps[iT,:]
    #         probs = model.model(Temp).numpy()    
    #         freq = np.round(probs*1000000).astype('int')
    #         freqlist = list(freq)
    #         freqs = arc.CheckedFrequencyTable(arc.SimpleFrequencyTable(freqlist )) 
    #         symb = ac_model.decode_symbol(freqs)
    #         ##TODO:construct the new template using the decoded symb:
    #         Temp[0:T12]    
            
    # probs = model.model(Temp).numpy()[0,:]
    # freqlist = list(np.round(probs*1000000).astype('int'))
    # # freqlist = list(probs)
    # # prob = probs[symb]
    # freqs = arc.CheckedFrequencyTable(arc.SimpleFrequencyTable(freqlist ))
    # ac_model.encode_symbol(freqs,symb)
    
    # if iT%10000==0:
    #     print(str(iT) + '/' + str(nTemps) )
        
        
        
        
        

def CodingCross_with_nn_probs(TempT22,DesT22,model,batch_size):

    # nb = TempT22.shape[1]
    mb = TempT22.shape[0]
    # nb1 = nb-1
    # NB2 = 2**nb
    
    CL =0
    n_zero_probs=0

    n_batches = np.ceil(mb/batch_size).astype('int')
    for i_batch in range(n_batches):
        symbs = DesT22[i_batch*batch_size:(i_batch+1)*batch_size]
        Conts = TempT22[i_batch*batch_size:(i_batch+1)*batch_size,:]
        probs = model.model(Conts).numpy()
        for icont in range(probs.shape[0]):
            prob = probs[icont,symbs[icont]]
            if prob==0:
                prob=0.001
                print("zero prob")
                n_zero_probs+=1
            CL+=  -np.log2(prob)
        if i_batch%100==0:
            print(str(i_batch) + '/' + str(n_batches))
    return CL,n_zero_probs



def Coding_with_nn_and_counts(TempT22,DesT22,model):


    
    counts = np.ones((2**22,2),'int32')
    mb = TempT22.shape[0]

    CL =0
    CL_ctx = 0
    n_zero_probs=0
    batch_size = 1000
    n_batches = np.ceil(mb/batch_size).astype('int')
    for i_batch in range(n_batches):
        symbs = DesT22[i_batch*batch_size:(i_batch+1)*batch_size]
        Conts = TempT22[i_batch*batch_size:(i_batch+1)*batch_size,:]
        multiplier = np.transpose(2**np.array(range(22))) 
        intConts = np.matmul( Conts , multiplier)
        
        
        probs = model.model(Conts).numpy()
        for icont in range(probs.shape[0]):
            prob = probs[icont,symbs[icont]]
            cprob = counts[intConts[icont],symbs[icont]]/np.sum(counts[intConts[icont],:])
            counts[intConts[icont],symbs[icont]]+=1
            if prob==0:
                prob=0.001
                print("zero prob")
                n_zero_probs+=1
            CL+=  -np.log2(prob)
            CL_ctx+= -np.log2(cprob)
        if i_batch%100==0:
            print(str(i_batch) + '/' + str(n_batches))
    return CL,CL_ctx,n_zero_probs