#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:36:29 2021

@author: root
"""
import numpy as np
import sys
from pcloud_functs import pcshow,lowerResolution,inds2vol,vol2inds,dilate_Loc
from ac_functs import ac_model2
from runlength import RLED
sys.path.append('/home/emre/Documents/kodlar/Reference-arithmetic-coding-master/python/')
from usefuls import in1d_index,plt_imshow,write_ints,read_ints,write_bits,read_bits,dec2bin2,bin2dec2,ints2bs,bs2ints
import arithmeticcoding as arc
import time
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



def OneSectOctMask2( icPC, BWTrue, BWTrueM, BWTrue1, BWTrue2, BWTrue1M, SectSize, StartStopLengthM, ctx_type ,ENC=True,nn_model='dec',ac_model='dec_and_enc',Location='for_debug',sess=None,for_train=False):

    # global esymbs,symbs

    wsize = 5#np.sqrt(ctx_type//4).astype(int)#np.sqrt((ctx_type+0.5)*2/5).astype(int)
    b = 2#(wsize-1)//2
    
    [ix1,iz1] = np.where(np.ones((wsize,wsize)))
    
    # Tempaxa= [iz1-b,ix1-b]
    # T12size = wsize**2    
    # TCsize = (T12size-1)//2
    # %%



    # % Go over the possible points
   
    # TempCaus = [Tempaxa[0][0:TCsize],Tempaxa[1][0:TCsize]]
    # dispz = np.zeros((TCsize,),'int')
    # dispx = np.zeros((TCsize,),'int')
    # dispznc = np.zeros((TCsize+1,),'int')
    # dispxnc = np.zeros((TCsize+1,),'int')
    # for i1 in range( TCsize):

    #     dispz[i1] = Tempaxa[0][i1]
    #     dispx[i1] = Tempaxa[1][i1]
        
    # for i1 in range( TCsize,T12size):
    #     dispznc[i1-TCsize] = Tempaxa[0][i1]
    #     dispxnc[i1-TCsize] = Tempaxa[1][i1]        
        
    nT = StartStopLengthM[icPC,2]    
    Temp = np.zeros((nT,ctx_type),'bool')
    Tprobs = np.zeros((nT,2),'float')
    if for_train:
        Des = np.zeros((nT,),'int')


    # iBBr_in=0
    # Locp = np.zeros((StartStopLengthM[icPC,2],3),'int')
    # TCaus = np.zeros((TCsize,),'bool')
    # TNCaus = np.zeros((TCsize+1,),'bool')
    # psymbs = []
    if ENC or for_train:
        # iTemp1 = -1
        for iTemp1,iBB in enumerate(range( StartStopLengthM[icPC,0],StartStopLengthM[icPC,1]+1)):
            iz = globz.LocM[iBB,2]
            ix = globz.LocM[iBB,0]
    

            Temp[iTemp1,0:25] = BWTrue2[iz-b:iz+b+1,ix-b:ix+b+1].flatten('F')
            Temp[iTemp1,25:50] = BWTrue1[iz-b:iz+b+1,ix-b:ix+b+1].flatten('F')
            Temp[iTemp1,50:75] = BWTrueM[iz-b:iz+b+1,ix-b:ix+b+1].flatten('F')
            # for i1 in range( TCsize):
            #     TCaus[i1] = BWTrue[iz+dispz[i1], ix+dispx[i1]]
            # for i1 in range( TCsize+1):    
            #     TNCaus[i1] = BWTrueM[iz+dispznc[i1], ix+dispxnc[i1]]
    
            # Temp[iTemp1,2*T12size:(2*T12size+TCsize)] = TCaus
            # Temp[iTemp1,(2*T12size+TCsize):3*T12size] = TNCaus

            Temp[iTemp1,75:] = BWTrue1M[iz-b:iz+b+1,ix-b:ix+b+1].flatten('F')
            
        if not for_train:               

            Tprobs = sess.run(nn_model.output,feed_dict={nn_model.input:Temp})                                    

    # iTemp = -1
##########2nd loop##########################################
    for iTemp,iBB in enumerate(range( StartStopLengthM[icPC,0],StartStopLengthM[icPC,1]+1)):
        # iTemp+=1
        iz = globz.LocM[iBB,2]
        ix = globz.LocM[iBB,0]
        
        if not ENC:
            Temp[iTemp,0:25] = BWTrue2[iz-b:iz+b+1,ix-b:ix+b+1].flatten('F')
            Temp[iTemp,25:50] = BWTrue1[iz-b:iz+b+1,ix-b:ix+b+1].flatten('F')
            Temp[iTemp,50:75] = BWTrueM[iz-b:iz+b+1,ix-b:ix+b+1].flatten('F')
            # for i1 in range( TCsize):
            #     TCaus[i1] = BWTrue[iz+dispz[i1], ix+dispx[i1]]
            # for i1 in range( TCsize+1):    
            #     TNCaus[i1] = BWTrueM[iz+dispznc[i1], ix+dispxnc[i1]]

            # Temp[iTemp,2*T12size:(2*T12size+TCsize)] = TCaus
            # Temp[iTemp,(2*T12size+TCsize):3*T12size] = TNCaus
           
            Temp[iTemp,75:] = BWTrue1M[iz-b:iz+b+1,ix-b:ix+b+1].flatten('F')
            
        if for_train:
            Des[iTemp] = BWTrue[iz, ix]
            
        if ENC and not for_train :
            symb = BWTrue[iz, ix]
            probs = Tprobs[iTemp,:]
        if not ENC: #DECODER
    
            probs = sess.run(nn_model.output,feed_dict={nn_model.input:Temp[iTemp:(iTemp+1),:]})[0,:]
        
        if not for_train:
            freq = np.ceil(probs*(2**14)).astype('int')#+1
            freqlist = list(freq)
    
                
            freqs = arc.SimpleFrequencyTable(freqlist )
     
         
            
            if ENC:
                ac_model.encode_symbol(freqs,symb)
    
    
            else:#DECODER
                symb = ac_model.decode_symbol(freqs)    
                if symb:
                    globz.Loc[globz.iBBr,:] = [ix,icPC,iz] 
                    globz.iBBr +=1
                    
            BWTrueM[iz, ix] = symb
                

    if for_train:
        return Temp,Des
    
def get_temps_dests2(ctx_type,ENC=True,nn_model ='dec',ac_model='dec_and_enc',maxesL='dec_and_enc',sess=None,bs_dir='',save_SSL=True,level=0,ori_level=0,dSSLs=None,for_train=False):

    # gtLoc = np.copy(Loc)
        
    maxX = maxesL[0]  
    maxY = maxesL[1]
    maxZ = maxesL[2]

    lSSL = maxY+10

    SectSize = (maxZ,maxX)

    
    
    # %% Find sections in Loc
    StartStopLength = np.zeros((lSSL,3),dtype='int')    
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
        # np.save('SSL2.npy',{'SSL2':SSL2,'ncPC':ncPC})
        if level==ori_level:

            ssbits = ''
            for ssbit in SSL2:
                ssbits=ssbits+str(int(ssbit))

            RLED(ssbits[32:-9],lSSL-41,lSSL-41,1,bs_dir+'rSSL.dat')
    else:    
        
        SSL2 = dSSLs[level,:]
    
    ncPC = np.max(np.where(SSL2)[0])
    #     infodict = np.load('SSL2.npy',allow_pickle=True)[()]
    #     ncPC = infodict['ncPC'][()]
    #     SSL2 = infodict['SSL2']
    # %% Find sections in LocM
    

    StartStopLengthM = np.zeros((lSSL,3),'int')
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
    

    
    if for_train:
        Temps = np.zeros((nM7,ctx_type),'bool')
        Dests = np.zeros((nM7,),'bool')

    if not ENC:        
        # iBBr=0
        globz.Loc = np.zeros((nM7,3),'int')
        # symbs = []
    # global symbs


    iTT = 0
    for icPC in range(ncPC+1):
        
        if icPC%50==0:
            print('icPC:' + str(icPC))



        if (StartStopLengthM[icPC,2] > 0) & SSL2[icPC] :
            
            
                        # %% 0. Mark the TRUE points on BWTrue
            BWTrue = np.zeros( SectSize,'bool')     
            if ENC:
                for iBB in range(StartStopLength[icPC,0],StartStopLength[icPC,1]+1):    
                    BWTrue[globz.Loc[iBB,2], globz.Loc[iBB,0]] = 1
        
            # %% 0.1 Mark the PREVIOUS SECTION TRUE points on BWTrue
            
            BWTrue1 = np.zeros( SectSize,'bool')
            if(icPC > 0):
                if( StartStopLength[icPC-1,1] > 0 ):
                    for iBB in range(StartStopLength[icPC-1,0],StartStopLength[icPC-1,1]+1):
                        BWTrue1[ globz.Loc[iBB,2], globz.Loc[iBB,0]] = 1
            # %% 0.2 Mark the PREVIOUS_PREVIOUS SECTION TRUE points on BWTrue
            
            BWTrue2 = np.zeros( SectSize,'bool')
            if(icPC > 1):
                if( StartStopLength[icPC-2,0] > 0 ):
                    for iBB in range(StartStopLength[icPC-2,0],StartStopLength[icPC-2,1]+1):
                        BWTrue2[globz.Loc[iBB,2], globz.Loc[iBB,0] ] = 1

            BWTrue1M = np.zeros( SectSize,'bool')
            if(icPC < ncPC):
                if( StartStopLengthM[icPC+1,1] > 0 ):
                    for iBB in range(StartStopLengthM[icPC+1,0],StartStopLengthM[icPC+1,1]+1):

                            BWTrue1M[ globz.LocM[iBB,2], globz.LocM[iBB,0]] = 1


            BWTrueM = np.zeros( SectSize,'bool')
            if( StartStopLengthM[icPC,1] > 0 ):
                for iBB in range(StartStopLengthM[icPC,0],StartStopLengthM[icPC,1]+1):
                    # try:
                    BWTrueM[ globz.LocM[iBB,2], globz.LocM[iBB,0]] = 1
            
            iBBr_prev = globz.iBBr
            if for_train:
                Temp, Des = OneSectOctMask2(icPC, BWTrue, BWTrueM, BWTrue1, BWTrue2,  BWTrue1M, SectSize, StartStopLengthM,ctx_type,ENC,nn_model,ac_model,sess=sess,for_train=for_train)              
                Temps[ iTT:(iTT+Temp.shape[0]),:]  = Temp
                Dests[ iTT:(iTT+Temp.shape[0])]  = Des    
                iTT = iTT+Temp.shape[0]

            else:
                OneSectOctMask2(icPC, BWTrue, BWTrueM, BWTrue1, BWTrue2, BWTrue1M, SectSize, StartStopLengthM,ctx_type,ENC,nn_model,ac_model,sess=sess)              

            iBBr_now = globz.iBBr
            # if ENC:   

            if not ENC:
                 # globz.symbs = globz.symbs+symbsp
                 # iBBr_in = Locp.shape[0]
                 iBBr_in = iBBr_now-iBBr_prev
                 if iBBr_in>0:
                     

                      StartStopLength[icPC,0] = iBBr_prev
                      StartStopLength[icPC,1] = iBBr_now-1
                      StartStopLength[icPC,2] = iBBr_now-iBBr_prev                    

                     

    if ENC and level==ori_level and not for_train:        

        freqlist = [10,10]
        freqs = arc.CheckedFrequencyTable(arc.SimpleFrequencyTable(freqlist )) 
        for i_s in range(64):
            ac_model.encode_symbol(freqs,0)
    
    if ENC:
        dec_Loc = 0
    if not(ENC) and not(for_train):
        dec_Loc =  globz.Loc[0:iBBr_now,:]

    if for_train:
        return Temps,Dests
    if not(for_train):
        return dec_Loc
    
        

def get_uctxs_counts2(GT,ctx_type,do_try_catch):


    
    # try:
        Location = GT -np.min(GT ,0)+32
        maxesL = np.max(Location,0)+[80,0,80]
        LocM = N_BackForth(Location )
        LocM_ro = np.unique(LocM[:,[1,0,2]],axis=0)
        LocM[:,[1,0,2]] = LocM_ro
        del LocM_ro
        globz.LocM = LocM        
        Loc_ro = np.unique(Location[:,[1,0,2]],axis=0)
        Location[:,[1,0,2]] = Loc_ro
        globz.Loc = Location
        del Loc_ro
        TempT,DesT = get_temps_dests2(ctx_type,nn_model =0,ac_model=0,maxesL=maxesL,for_train=True)
    
    
        uctxs,ic=np.unique(TempT,return_inverse=True,axis=0)
    
        nuctxs = uctxs.shape[0]
        counts = np.zeros((nuctxs,2),'int')
    
        for i1 in range(DesT.shape[0]):#=1:size(DesT,1)
            symb = int(DesT[i1])
            counts[ic[i1],symb]=counts[ic[i1],symb]+1
    

        return uctxs,counts
    
 
def ENCODE_DECODE(ENC,bs_dir,nn_model,ac_model,sess,ori_level=0,GT=0):
    
    start = time.time()      


    nintbits = ori_level*np.ones((6,),int)
    lrGTs = dict()
    if ENC:# or debug_dec:

        minsG = np.min(GT ,0)
        maxesG = np.max(GT,0)
        minmaxesG = np.concatenate((minsG,maxesG))
        
        # write_ints(minmaxesG,nintbits,bs_dir+'maxes_mins.dat')    
        sibs = ints2bs(minmaxesG,nintbits)
        
        lrGTs[ori_level] = GT
        lrGT = np.copy(GT)
        for il in range(ori_level-2):
            lrGT = lowerResolution(lrGT)
            lrGTs[ori_level-il-1] = lrGT
            
        lowest_bs = inds2vol(lrGTs[2],[4,4,4]).flatten().astype(int)
        lowest_str = ''
        for ibit in range(64):
            lowest_str = lowest_str+str(lowest_bs[ibit])
            
        #write_bits(lowest_str+'1',bs_dir+'lowest.dat')
        sibs = sibs+lowest_str+'1'
        write_bits(sibs,bs_dir+'side_info.bs')
        dSSLs = 0
    if not ENC:
        
        # minmaxesG =read_ints(nintbits,bs_dir+'maxes_mins.dat')
        # lowest_str = read_bits(bs_dir+'lowest.dat')[0:64]
        sibs = read_bits(bs_dir+'side_info.bs')
        minmaxesG = bs2ints(sibs[0:np.sum(nintbits)],nintbits)
        lowest_str = sibs[np.sum(nintbits):-1]
        
        lowest_bs = np.zeros([64,],int)
        for ibit in range(64):
            lowest_bs[ibit] = int(lowest_str[ibit])
        vol = lowest_bs.reshape([4,4,4])
        lrGTs[2] = vol2inds(vol)

     
        
        lrmm = np.copy(minmaxesG[np.newaxis,:])
        lrmms=np.zeros((ori_level+1,6),int)
        lrmms[ori_level] = lrmm
        for il in range(ori_level-2):
            lrmm = lowerResolution(lrmm)
            lrmms[ori_level-il-1,:] = lrmm
            
        mins11 = lrmms[ori_level,1]
        maxes11 = lrmms[ori_level,4]        
        lSSL = maxes11-mins11+32+10
                
       ##get dssls     
        dSSLs  = np.zeros((ori_level+1,4000),int)
        #ssbits = read_bits(bs_dir+'SSL.dat')[:-1]#[(ori_level+2):-1]
        ssbits = 32*'0'
        ssbits = ssbits + RLED('',lSSL-41,lSSL-41,0,bs_dir+'rSSL.dat') +9*'0'
        # dSSL = ssbits    
        for ib,bit in enumerate(ssbits):
            dSSLs[ori_level,ib] = int(bit) 
            
        # dncPCs = np.zeros((ori_level+1),int)
        # dncPCs[ori_level] = np.max(np.where(dSSLs[ori_level])[0])
        for level in range(ori_level,3,-1):
            # lrmms[level][1]%2
            add = lrmms[level][1]%2#1-np.where(SSLs[level])[0][-1]%2
            inds = lowerResolution(np.where(dSSLs[level])[0]+add-32)+32
            dSSLs[level-1,inds] = 1
            # dncPCs[level-1] = np.max(np.where(dSSLs[level-1])[0])
            # print(level)        

    for level in range(3,ori_level+1):
        
        # globz.isymb = 0
        globz.iBBr = 0

        
        if ENC: #or debug_dec:
            mins1 = np.min(lrGTs[level] ,0)
            maxes1 = np.max(lrGTs[level],0)
            Location = lrGTs[level] -mins1+32
    
        if not ENC:
            mins1 = lrmms[level,0:3]
            maxes1 = lrmms[level,3:6]
        
        maxesL = maxes1-mins1+32+[80,0,80]
    
        LocM = dilate_Loc(lrGTs[level-1])-mins1+32
        
        LocM_ro = np.unique(LocM[:,[1,0,2]],axis=0)
        LocM[:,[1,0,2]] = LocM_ro
        del LocM_ro
        globz.LocM = LocM
    
    
        if ENC :#or debug_dec:
            Loc_ro = np.unique(Location[:,[1,0,2]],axis=0)
            Location[:,[1,0,2]] = Loc_ro
            globz.Loc = Location
            del Loc_ro

        dec_Loc= get_temps_dests2(nn_model.ctx_type,ENC,nn_model = nn_model,ac_model=ac_model,maxesL = maxesL,sess=sess,bs_dir=bs_dir,save_SSL=True,level=level,ori_level=ori_level,dSSLs=dSSLs)
    
    
        if not ENC:
            lrGTs[level] = dec_Loc+mins1-32
    
    
    if ENC:
        ac_model.end_encoding()
    
    end = time.time()
    time_spent = end - start
    nmins = int(time_spent//60)
    nsecs = int(np.round(time_spent-nmins*60))
    print('time spent: ' + str(nmins) + 'm ' + str(nsecs) + 's')
    
    
    if not ENC:
        dec_GT = lrGTs[level]
    else:
        dec_GT = 0
    
    return dec_GT,time_spent
    

    

      