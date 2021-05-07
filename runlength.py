#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:11:04 2021

@author: root
"""

import numpy as np
from usefuls import ints2bs,bs2ints,write_bits,read_bits,bin2dec2,dec2bin2
import os



def RLED(bits,maxrunl,maxnruns,ENC,fpath):
    nbits_needed = int(np.ceil(np.log(maxrunl+1)/np.log(2)))
    nnrun_bits = int(np.ceil(np.log(maxnruns+1)/np.log(2)))


    
    if ENC:
        start_bit = bits[0]
    else:
        bs = read_bits(fpath)
        start_bit = bs[0]
        nruns = bin2dec2(bs[1:(nnrun_bits+1)])
        nintbits = nbits_needed*np.ones((nruns,),int)
        runls = bs2ints(bs[(nnrun_bits+1):],nintbits)        
        print('dec runs:')
        print(runls)
        bits = ''
        curr_bit=start_bit
        for irun,runl in enumerate(runls):
            bits = bits + runl*curr_bit
            curr_bit=str(1-int(curr_bit))
            
        return bits
        
        
        
        
    last_bit = start_bit
    curr_run_start = 0
    if ENC:
        runls = np.zeros((len(bits)+1,),int)
    
        irun=0
        for ibit,bit in enumerate(bits[1:]):
        
            if bit!=last_bit:       
                runls[irun] = ibit+1-curr_run_start
                curr_run_start = ibit+1                
                irun +=1
                last_bit=bit
        
        runls[irun] = ibit+2-curr_run_start
        runls = runls[0:(irun+1)]
        print('enc runs:')
        print(runls)
        nruns = irun+1
        bs1 = dec2bin2(nruns,nnrun_bits)
        nintbits = nbits_needed*np.ones((nruns,),int)
        bs = start_bit + bs1 + ints2bs(runls,nintbits)
        write_bits(bs,fpath)
    


if __name__=="__main__":
    bits = '011111000000000'
    fpath = 'bs.bs'
    print('encoded bits: ' + bits)
    RLED(bits, len(bits), len(bits), 1, fpath)
    dbits = RLED('', len(bits), len(bits), 0, fpath)
    print('decoded bits: ' + dbits)
    CL = os.path.getsize(fpath)
    print('bitrate:'+str(CL/len(bits)))
    
    
