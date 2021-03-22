#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:36:29 2021

@author: root
"""
import numpy as np
import sys
from pcloud_functs import pcshow
sys.path.append('/home/emre/Documents/kodlar/Reference-arithmetic-coding-master/python/')
from usefuls import in1d_index,plt_imshow
import arithmeticcoding as arc

import globz

def N_BackForth(sBBi): ##checked with matlab output

# %% Move from current resolution one level down and then one up
# %% In the UP step enforce at each cube all 8 possible patterns  
# %% input: sBBi, the sorted PC, by unique

    quotBB = np.floor(sBBi/2).astype('int')                    #% size is (nBBx3)
    Points_parent,iC = np.unique(quotBB,return_inverse=True,axis=0)  #   % size of iC is (nBBx3)

    PatEl = np.array( [[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1]])
                       

    
    BlockiM = np.zeros([0,3],'int')
    for iloc in range(8):    # % size(PatEl,1) = 8
        iv3 = PatEl[iloc,:]
        Blocki = Points_parent*2+iv3
        BlockiM = np.concatenate((BlockiM,Blocki),0) 

    LocM = np.unique(BlockiM,axis=0)
    return LocM



def OneSectOctMask2( icPC, BWTrue, BWTrue1, BWTrue2, SectSize, StartStopLengthM, ctx_type ,ENC=True,nn_model='dec',ac_model='dec_and_enc',Location='for_debug'):

    # global esymbs,symbs

    wsize = np.sqrt((ctx_type+0.5)*2/5).astype(int)
    b = (wsize-1)//2
    
    [ix1,iz1] = np.where(np.ones((wsize,wsize)))
    
    Tempaxa= [iz1-b,ix1-b]
    T12size = wsize**2    
    TCsize = (T12size-1)//2
    # %%



    # % Go over the possible points
    # iTemp = 0
    # TempCaus = [Tempaxa[0][0:TCsize],Tempaxa[1][0:TCsize]]
    dispz = np.zeros((TCsize,),'int')
    dispx = np.zeros((TCsize,),'int')
    for i1 in range( TCsize):

        dispz[i1] = Tempaxa[0][i1]
        dispx[i1] = Tempaxa[1][i1]
        
    nT = StartStopLengthM[icPC,2]    
    # Temp = np.zeros((nT,ctx_type),'int')
    # Des = np.zeros((nT,),'int')



    TCaus = np.zeros((TCsize,),'int')

    for iBB in range( StartStopLengthM[icPC,0],StartStopLengthM[icPC,1]+1):
        iz = globz.LocM[iBB,2]
        ix = globz.LocM[iBB,0]

        Temp2 = BWTrue2[(iz-b):(iz+b+1),(ix-b):(ix+b+1)].flatten('F') #(2b+1)**2
        Temp1 = BWTrue1[(iz-b):(iz+b+1),(ix-b):(ix+b+1)].flatten('F')
    
        for i1 in range( TCsize):
            TCaus[i1] = BWTrue[iz+dispz[i1], ix+dispx[i1]]

                    

        
        globz.Temps[globz.iTemp,0:T12size] = Temp2
        globz.Temps[globz.iTemp,T12size:2*T12size] = Temp1
        globz.Temps[globz.iTemp,2*T12size:ctx_type] = TCaus
        
        if ENC:
            globz.Desds[globz.iTemp] = BWTrue[iz, ix]

            symb = globz.Desds[globz.iTemp]
            if symb:
                globz.iBBr +=1
        if not ENC:
            probs = nn_model.model(globz.Temps[globz.iTemp:(globz.iTemp+1),:]).numpy()[0,:]       
            freq = np.round(probs*1000000).astype('int')
            freqlist = list(freq)
            freqs = arc.CheckedFrequencyTable(arc.SimpleFrequencyTable(freqlist )) 
            symb = ac_model.decode_symbol(freqs)  

            globz.isymb +=1
            

            BWTrue[iz, ix] = symb
            if symb:

                globz.Loc[globz.iBBr,:] = [ix,icPC,iz] 

                globz.iBBr +=1
        
        globz.iTemp+=1        
    if ENC:
        # bsize = globz.batch_size) 
        nb = np.ceil(nT/globz.batch_size).astype('int')
        
        for ib in range(nb):
            bsymbs = globz.Desds[ib*globz.batch_size:(ib+1)*globz.batch_size]
            bTemps = globz.Temps[ib*globz.batch_size:(ib+1)*globz.batch_size,:]            
            bprobs = nn_model.model(bTemps).numpy()    
            # bfreqs = np.round(bprobs*1000000).astype('int')
            
            for iT in range(len(bsymbs)):
                freq = np.round(bprobs[iT,:]*1000000).astype('int')
                symb = bsymbs[iT]
                freqlist = list(freq)
                freqs = arc.CheckedFrequencyTable(arc.SimpleFrequencyTable(freqlist ))        
                ac_model.encode_symbol(freqs,symb)
                globz.isymb +=1        
                           
                       

    # if not ENC:

    #     Des = 0
        
    # return Temp,Des

def get_temps_dests2(ctx_type,ENC=True,nn_model ='dec',ac_model='dec_and_enc',maxesL='dec_and_enc'):

        
    maxX = maxesL[0]  
    maxY = maxesL[1]
    maxZ = maxesL[2]



    SectSize = (maxZ,maxX)
    
    # %% Find sections in Loc
    StartStopLength = np.zeros((maxY+40,3),dtype='int')    
    if ENC:
        icPC = globz.Loc[0,1]    
        StartStopLength[icPC,0] = 0
        for iBB in range(globz.Loc.shape[0]):#= 1:(size(Loc,1))
            if(globz.Loc[iBB,1] > icPC):
                StartStopLength[icPC,1] = iBB-1
                StartStopLength[icPC,2] = StartStopLength[icPC,1]-StartStopLength[icPC,0]+1
                icPC = globz.Loc[iBB,1]
                StartStopLength[icPC,0] = iBB
    
        iBB = globz.Loc.shape[0]
        if(globz.Loc[iBB-1,1] == icPC):
            StartStopLength[icPC,1] = iBB-1
            StartStopLength[icPC,2] = StartStopLength[icPC,1]-StartStopLength[icPC,0]+1

        ncPC = np.copy(icPC)
        SSL2 = StartStopLength[:,2]>0
        np.save('SSL2.npy',{'SSL2':SSL2,'ncPC':ncPC})
    else:    
        
        infodict = np.load('SSL2.npy',allow_pickle=True)[()]
        ncPC = infodict['ncPC'][()]
        SSL2 = infodict['SSL2']
    # %% Find sections in LocM
    

    StartStopLengthM = np.zeros((maxY+40,3),'int')
    icPC = globz.LocM[0,1]
    StartStopLengthM[icPC,0] = 0
    for iBB in range(globz.LocM.shape[0]):
        if(globz.LocM[iBB,1] > icPC):
            StartStopLengthM[icPC,1] = iBB-1
            StartStopLengthM[icPC,2] = StartStopLengthM[icPC,1]-StartStopLengthM[icPC,0]+1
            icPC = globz.LocM[iBB,1]
            StartStopLengthM[icPC,0] = iBB

    iBB = globz.LocM.shape[0]
    if(globz.LocM[iBB-1,1] == icPC):
        StartStopLengthM[icPC,1] = iBB-1
        StartStopLengthM[icPC,2] = StartStopLengthM[icPC,1]-StartStopLengthM[icPC,0]+1

    

    
    
    # %%
    # nM = np.max(LocM[:,1])
    nM7 = globz.LocM.shape[0]
    

    
    # if ENC:
    globz.Temps = np.zeros((nM7,ctx_type),'int')
    globz.Desds = np.zeros((nM7,),'int')

    if not ENC:      
        # iBBr=0
        globz.Loc = np.zeros((nM7,3),'int')


    # iTT = 0
    for icPC in range(ncPC+1):
        
        if icPC%50==0:
            print('icPC:' + str(icPC))


        if (StartStopLengthM[icPC,2] > 0) & SSL2[icPC] :
            
                        # %% 0. Mark the TRUE points on BWTrue
            BWTrue = np.zeros( SectSize,'int')     
            if ENC:
                for iBB in range(StartStopLength[icPC,0],StartStopLength[icPC,1]+1):    
                    BWTrue[globz.Loc[iBB,2], globz.Loc[iBB,0]] = 1
        
            # %% 0.1 Mark the PREVIOUS SECTION TRUE points on BWTrue
            
            BWTrue1 = np.zeros( SectSize,'int')
            if(icPC > 0):
                if( StartStopLength[icPC-1,1] > 0 ):
                    for iBB in range(StartStopLength[icPC-1,0],StartStopLength[icPC-1,1]+1):
                        BWTrue1[ globz.Loc[iBB,2], globz.Loc[iBB,0]] = 1
            # %% 0.2 Mark the PREVIOUS_PREVIOUS SECTION TRUE points on BWTrue
            
            BWTrue2 = np.zeros( SectSize,'int')
            if(icPC > 1):
                if( StartStopLength[icPC-2,0] > 0 ):
                    for iBB in range(StartStopLength[icPC-2,0],StartStopLength[icPC-2,1]+1):
                        BWTrue2[globz.Loc[iBB,2], globz.Loc[iBB,0] ] = 1

            
            iBBr_prev = globz.iBBr
            # Temp, Des,Locp =  OneSectOctMask2(icPC, BWTrue, BWTrue1, BWTrue2, LocM, SectSize, StartStopLengthM,ctx_type)
            OneSectOctMask2(icPC, BWTrue, BWTrue1, BWTrue2, SectSize, StartStopLengthM,ctx_type,ENC,nn_model,ac_model)              

            iBBr_now = globz.iBBr
            # if ENC:   

            if not ENC:

                 iBBr_in = iBBr_now-iBBr_prev
                 if iBBr_in>0:

                      StartStopLength[icPC,0] = iBBr_prev
                      StartStopLength[icPC,1] = iBBr_now-1
                      StartStopLength[icPC,2] = iBBr_now-iBBr_prev                    

    if ENC:        
        # return Temps,Dests    
        freqlist = [10,10]
        freqs = arc.CheckedFrequencyTable(arc.SimpleFrequencyTable(freqlist )) 
        for i_s in range(32):
            ac_model.encode_symbol(freqs,0)
        
        dec_Loc = 0
    else:
        dec_Loc =  globz.Loc[0:iBBr_now,:]
    Temps = 0
    Dests  = 0

    return Temps,Dests,dec_Loc
    
        



      