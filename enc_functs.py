#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:36:29 2021

@author: root
"""
import numpy as np




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



def OneSectOctMask2( icPC,  BB007, BB007M, SectSize, StartStopLength, StartStopLengthM, ctx_type ):


    wsize = np.sqrt((ctx_type+0.5)*2/5).astype(int)
    b = (wsize-1)//2
    
    [ir,ic] = np.where(np.ones((wsize,wsize)))
    
    Tempaxa= [ic-b,ir-b]
    # %%
    # % x = ismember(StartStopLength,[0 0 0],'rows');
    # % inds = find(diff(x));
    # % StartStopLength=StartStopLength(1:inds(2),:);
    # % StartStopLengthM=StartStopLengthM(1:inds(2),:);

# %% 0. Mark the TRUE points on BWTrue
    BWTrue = np.zeros( (SectSize[0], SectSize[1]),'int8')
    for iBB in range(StartStopLength[icPC,0],StartStopLength[icPC,1]+1):

        BWTrue[BB007[iBB,2], BB007[iBB,0]] = 1




# %% 0.1 Mark the PREVIOUS SECTION TRUE points on BWTrue

    BWTrue1 = np.zeros( (SectSize[0], SectSize[1] ),'int8')
    if(icPC > 0):
        if( StartStopLength[icPC-1,1] > 0 ):
            for iBB in range(StartStopLength[icPC-1,0],StartStopLength[icPC-1,1]+1):
                BWTrue1[ BB007[iBB,2], BB007[iBB,0]] = 1


# %% 0.2 Mark the PREVIOUS_PREVIOUS SECTION TRUE points on BWTrue

    BWTrue2 = np.zeros( (SectSize[0], SectSize[1]),'int8')
    if(icPC > 1):
        if( StartStopLength[icPC-2,0] > 0 ):
            for iBB in range(StartStopLength[icPC-2,0],StartStopLength[icPC-2,1]+1):
                BWTrue2[BB007[iBB,2], BB007[iBB,0] ] = 1



    #%% 0. Mark the Masked points from previous resolution
        # BW_Masked np.zeros( (SectSize[0]+2*pad, SectSize[1]+2*pad ))
        # for iBB in range(StartStopLengthM[icPC,0],StartStopLengthM[icPC,1]+1):
        #     BW_Masked[ BB007M[iBB,2], BB007M[iBB,0] ] = 1
    T12size = wsize**2    
    TCsize = (T12size-1)//2
    # % Go over the possible points
    iTemp = -1
    TempCaus = [Tempaxa[0][0:TCsize],Tempaxa[1][0:TCsize]]
    disp0 = np.zeros((TCsize,),'int8')
    disp1 = np.zeros((TCsize,),'int8')
    for i1 in range( TCsize):

        disp0[i1] = TempCaus[0][i1]
        disp1[i1] = TempCaus[1][i1]
        
    Temp = np.zeros((StartStopLengthM[icPC,2],ctx_type),'int8')
    Des = np.zeros((StartStopLengthM[icPC,2],),'int8')

    
    
    
    for iBB in range( StartStopLengthM[icPC,0],StartStopLengthM[icPC,1]+1):
        ir = BB007M[iBB,2]
        ic = BB007M[iBB,0]

        Temp2 = BWTrue2[(ir-b):(ir+b+1),(ic-b):(ic+b+1)].flatten('F') #(2b+1)**2
        Temp1 = BWTrue1[(ir-b):(ir+b+1),(ic-b):(ic+b+1)].flatten('F')
    
        TCaus = np.zeros((TCsize,),'int8')
        for i1 in range( TCsize):
            # try:
            # TCaus[i1] = BWTrue[ir+TempCaus[0][i1]+pad, ic+TempCaus[1][i1]+pad]
            TCaus[i1] = BWTrue[ir+disp0[i1], ic+disp1[i1]]
            # except:
            #     sth_wrong=1
                    
        iTemp = iTemp +1
        # Temp[iTemp, 0:ctx_type] = [Temp2(:); Temp1(:); TCaus(:)]'
        # Temp[iTemp,0:ctx_type] = np.concatenate((Temp2,Temp1,TCaus))
        Temp[iTemp,0:T12size] = Temp2
        Temp[iTemp,T12size:2*T12size] = Temp1
        Temp[iTemp,2*T12size:ctx_type] = TCaus
        Des[iTemp] = BWTrue[ir, ic]
        
        # temp_good = np.prod(TempTm[iTemp,0:ctx_type] == Temp[iTemp,0:ctx_type])
        
        # temp_not_good =0
        # if not(temp_good):
        #     temp_not_good = 1
        

    return Temp,Des

def get_temps_dests2(BB007,ctx_type):


    # %% Pick the resolution and split to small brick
    BB007=BB007+8

    BB007M = N_BackForth(BB007)
    
    BB007_ro = np.unique(BB007[:,[1,0,2]],axis=0)
    BB007[:,[1,0,2]] = BB007_ro
    del BB007_ro

    
    
    nrPC,ncPC,maxH = np.max(BB007,0)+10
    # ncPC = int(np.max(BB007[:,1])+10)
    # nrPC = int(np.max(BB007[:,0])+10)
    # maxH = int(np.max(BB007[:,2])+10)
    SectSize = (maxH,nrPC)

    
    
    # %% Find sections in BB007
    StartStopLength = np.zeros((ncPC,3),dtype='int')
    icPC = BB007[0,1]
    StartStopLength[icPC,0] = 0
    for iBB in range(BB007.shape[0]):#= 1:(size(BB007,1))
        if(BB007[iBB,1] > icPC):
            StartStopLength[icPC,1] = iBB-1
            StartStopLength[icPC,2] = StartStopLength[icPC,1]-StartStopLength[icPC,0]+1
            icPC = BB007[iBB,1]
            StartStopLength[icPC,0] = iBB

    iBB = BB007.shape[0]
    if(BB007[iBB-1,1] == icPC):
        StartStopLength[icPC,1] = iBB-1
        StartStopLength[icPC,2] = StartStopLength[icPC,1]-StartStopLength[icPC,0]+1

 
    ##BURADAN DEVAM!!!!!!!!!!!
    # %% Find sections in BB007M
    

    BB007M_ro = np.unique(BB007M[:,[1,0,2]],axis=0)
    BB007M[:,[1,0,2]] = BB007M_ro
    del BB007M_ro
    StartStopLengthM = np.zeros((ncPC,3),'int')
    icPC = BB007M[0,1]
    StartStopLengthM[icPC,0] = 0
    for iBB in range(BB007M.shape[0]):
        if(BB007M[iBB,1] > icPC):
            StartStopLengthM[icPC,1] = iBB-1
            StartStopLengthM[icPC,2] = StartStopLengthM[icPC,1]-StartStopLengthM[icPC,0]+1
            icPC = BB007M[iBB,1]
            StartStopLengthM[icPC,0] = iBB

    iBB = BB007M.shape[0]
    if(BB007M[iBB-1,1] == icPC):
        StartStopLengthM[icPC,1] = iBB-1
        StartStopLengthM[icPC,2] = StartStopLengthM[icPC,1]-StartStopLengthM[icPC,0]+1

    

    
    
    # %%
    nM = np.max(BB007M[:,1])
    
    Temps = np.zeros((BB007M.shape[0],ctx_type),'int8')
    Dests = np.zeros((BB007M.shape[0]),'int8')
    


    iTT = 0
    for icPC in range(nM):
        print('icPC:' + str(icPC))

        if (StartStopLengthM[icPC,2] > 0) & (StartStopLength[icPC,2] > 0) :
            

            Temp, Des =  OneSectOctMask2(icPC,  BB007, BB007M, SectSize, StartStopLength, StartStopLengthM,ctx_type)
            Temps[ iTT:(iTT+Temp.shape[0]),0:ctx_type]  = Temp
            Dests[ iTT:(iTT+Temp.shape[0])]  = Des
            

            
            iTT = iTT+Temp.shape[0]

            
    return Temps,Dests    


    
        



      