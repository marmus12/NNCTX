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

import arithmeticcoding as arc



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

    uBlockiM = np.unique(BlockiM,axis=0)
    return uBlockiM



def OneSectOctMask2( icPC, BWTrue, BWTrue1, BWTrue2, LocM, SectSize, StartStopLengthM, ctx_type ,ENC=True,nn_model='dec',ac_model='dec',iBBr ='dec'):



    wsize = np.sqrt((ctx_type+0.5)*2/5).astype(int)
    b = (wsize-1)//2
    
    [ir,ic] = np.where(np.ones((wsize,wsize)))
    
    Tempaxa= [ic-b,ir-b]
    T12size = wsize**2    
    TCsize = (T12size-1)//2
    # %%



    # % Go over the possible points
    iTemp = -1
    TempCaus = [Tempaxa[0][0:TCsize],Tempaxa[1][0:TCsize]]
    disp0 = np.zeros((TCsize,),'int')
    disp1 = np.zeros((TCsize,),'int')
    for i1 in range( TCsize):

        disp0[i1] = TempCaus[0][i1]
        disp1[i1] = TempCaus[1][i1]
        
    nT = StartStopLengthM[icPC,2]    
    Temp = np.zeros((nT,ctx_type),'int')
    Des = np.zeros((nT,),'int')


    iBBr_in=-1
    Locp = np.zeros((StartStopLengthM[icPC,2],3),'int')
    for iBB in range( StartStopLengthM[icPC,0],StartStopLengthM[icPC,1]+1):
        ir = LocM[iBB,2]
        ic = LocM[iBB,0]

        Temp2 = BWTrue2[(ir-b):(ir+b+1),(ic-b):(ic+b+1)].flatten('F') #(2b+1)**2
        Temp1 = BWTrue1[(ir-b):(ir+b+1),(ic-b):(ic+b+1)].flatten('F')
    
        TCaus = np.zeros((TCsize,),'int')
        for i1 in range( TCsize):

            TCaus[i1] = BWTrue[ir+disp0[i1], ic+disp1[i1]]

                    
        iTemp+=1
        
        Temp[iTemp,0:T12size] = Temp2
        Temp[iTemp,T12size:2*T12size] = Temp1
        Temp[iTemp,2*T12size:ctx_type] = TCaus
        if ENC:
            Des[iTemp] = BWTrue[ir, ic]
        else:
            probs = nn_model.model(Temp[iTemp:(iTemp+1),:]).numpy()[0,:]
            freq = np.round(probs*1000000).astype('int')
            freqlist = list(freq)
            freqs = arc.CheckedFrequencyTable(arc.SimpleFrequencyTable(freqlist )) 
            symb = ac_model.decode_symbol(freqs)           
            # Des[iTemp] = symb 
            BWTrue[ir, ic] = symb
            if symb:
                iBBr_in +=1
                Locp[iBBr_in,0] = ic                 
                Locp[iBBr_in,1] = icPC 
                Locp[iBBr_in,2] = ir

                
                
                

    if ENC:
        return Temp,Des
    else:
        return Locp[0:iBBr_in,:]

def get_temps_dests2(Loc,ctx_type,ENC=True,LocM='dec',nn_model ='dec',ac_model='dec',maxesL='dec'):

    if ENC:
        # %% Pick the resolution and split to small brick
    
        LocM = N_BackForth(Loc)
        
        Loc_ro = np.unique(Loc[:,[1,0,2]],axis=0)
        Loc[:,[1,0,2]] = Loc_ro
        del Loc_ro
    
        
        
        nrPC,ncPC,maxH = np.max(Loc,0)+10
        # ncPC = int(np.max(Loc[:,1])+10)
        # nrPC = int(np.max(Loc[:,0])+10)
        # maxH = int(np.max(Loc[:,2])+10)
        SectSize = (maxH,nrPC)
    else:
        # aaaaa=5
        ncPC = maxesL[1]+10
        SectSize = (maxesL[2]+10,maxesL[0]+10)
        #maxH,nrPC = SectSize
        # TODO: read from file maxH,nrPC,ncPC,StartStopLength
    
    
    # %% Find sections in Loc
    StartStopLength = np.zeros((ncPC,3),dtype='int')    
    if ENC:
        # icPC0 = Loc[0,1] 
        #TODO: write to file:maxH,nrPC,ncPC,icPC0
        icPC = Loc[0,1] 
    
        StartStopLength[icPC,0] = 0
        for iBB in range(Loc.shape[0]):#= 1:(size(Loc,1))
            if(Loc[iBB,1] > icPC):
                StartStopLength[icPC,1] = iBB-1
                StartStopLength[icPC,2] = StartStopLength[icPC,1]-StartStopLength[icPC,0]+1
                icPC = Loc[iBB,1]
                StartStopLength[icPC,0] = iBB
    
        iBB = Loc.shape[0]
        if(Loc[iBB-1,1] == icPC):
            StartStopLength[icPC,1] = iBB-1
            StartStopLength[icPC,2] = StartStopLength[icPC,1]-StartStopLength[icPC,0]+1

 
    ##BURADAN DEVAM!!!!!!!!!!!
    # %% Find sections in LocM
    
    LocM_ro = np.unique(LocM[:,[1,0,2]],axis=0)
    LocM[:,[1,0,2]] = LocM_ro
    del LocM_ro
    StartStopLengthM = np.zeros((ncPC,3),'int')
    icPC = LocM[0,1]
    StartStopLengthM[icPC,0] = 0
    for iBB in range(LocM.shape[0]):
        if(LocM[iBB,1] > icPC):
            StartStopLengthM[icPC,1] = iBB-1
            StartStopLengthM[icPC,2] = StartStopLengthM[icPC,1]-StartStopLengthM[icPC,0]+1
            icPC = LocM[iBB,1]
            StartStopLengthM[icPC,0] = iBB

    iBB = LocM.shape[0]
    if(LocM[iBB-1,1] == icPC):
        StartStopLengthM[icPC,1] = iBB-1
        StartStopLengthM[icPC,2] = StartStopLengthM[icPC,1]-StartStopLengthM[icPC,0]+1

    

    
    
    # %%
    nM = np.max(LocM[:,1])
    nM7 = LocM.shape[0]
    

    
    if ENC:
        Temps = np.zeros((nM7,ctx_type),'int')
        Dests = np.zeros((nM7,),'int')
        # nn_model = ac_model=iBBr =0
    else:        
        iBBr=0
        Loc = np.zeros((5000000,3),'int')


    iTT = 0
    for icPC in range(nM):
        
        if icPC%50==0:
            print('icPC:' + str(icPC))
        if ENC:
            condition = (StartStopLengthM[icPC,2] > 0) & (StartStopLength[icPC,2] > 0)
        else:
            condition = (StartStopLengthM[icPC,2] > 0)  # TODO: THIS TO BE CHECKED
            
        if condition :
            
            
                        # %% 0. Mark the TRUE points on BWTrue
            BWTrue = np.zeros( (SectSize[0], SectSize[1]),'int')     
            if ENC:
                for iBB in range(StartStopLength[icPC,0],StartStopLength[icPC,1]+1):    
                    BWTrue[Loc[iBB,2], Loc[iBB,0]] = 1
        
            # %% 0.1 Mark the PREVIOUS SECTION TRUE points on BWTrue
            
            BWTrue1 = np.zeros( (SectSize[0], SectSize[1] ),'int')
            if(icPC > 0):
                if( StartStopLength[icPC-1,1] > 0 ):
                    for iBB in range(StartStopLength[icPC-1,0],StartStopLength[icPC-1,1]+1):
                        BWTrue1[ Loc[iBB,2], Loc[iBB,0]] = 1
            # %% 0.2 Mark the PREVIOUS_PREVIOUS SECTION TRUE points on BWTrue
            
            BWTrue2 = np.zeros( (SectSize[0], SectSize[1]),'int')
            if(icPC > 1):
                if( StartStopLength[icPC-2,0] > 0 ):
                    for iBB in range(StartStopLength[icPC-2,0],StartStopLength[icPC-2,1]+1):
                        BWTrue2[Loc[iBB,2], Loc[iBB,0] ] = 1

            
            if ENC:
                Temp, Des =  OneSectOctMask2(icPC, BWTrue, BWTrue1, BWTrue2, LocM, SectSize, StartStopLengthM,ctx_type)
                
                Temps[ iTT:(iTT+Temp.shape[0]),0:ctx_type]  = Temp
                Dests[ iTT:(iTT+Temp.shape[0])]  = Des
    
                iTT = iTT+Temp.shape[0]
            else:
                 Locp =  OneSectOctMask2(icPC, BWTrue, BWTrue1, BWTrue2, LocM, SectSize, StartStopLengthM,ctx_type,ENC,nn_model,ac_model,iBBr)              
                 iBBr_in = Locp.shape[0]
                 if iBBr_in>0:
                     
                     Loc[iBBr:(iBBr+iBBr_in),:] = Locp
                     # if icPC%50==0:
                     #     pcshow(Loc)
                     StartStopLength[icPC,2] = iBBr_in
                     StartStopLength[icPC,0] = iBBr

                     iBBr+=iBBr_in
                     StartStopLength[icPC,1] = iBBr
                     
                     

    if ENC:        
        return Temps,Dests    
    else:
        return Loc


    
        



      