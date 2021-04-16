#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:56:15 2021

@author: root
"""
from usefuls import read_bits,write_bits,write_ints,read_ints
import numpy as np
import os
from dec2bin import dec2bin


ints2write = np.array([127,255,1023],'int')
nintbits = [7,8,10]



fpath = 'test.bits'
write_ints(ints2write,nintbits,fpath)



intsread = read_ints(nintbits,fpath)
assert(np.prod(intsread==ints2write))

filesize=os.path.getsize(fpath)*8
print('filesize:'+str(filesize))

rbits=read_bits(fpath)
#%%########




inbits = ''
for i,inte in enumerate(ints2write):
    intbin = dec2bin(inte)
    
    inbits=inbits+'0'*(nintbits[i]-len(intbin)) +intbin

inbits = inbits+'1'
# inbits = "00000000000000000000001001"
print('len(inbits):'+str(len(inbits)))

write_bits(inbits,fpath)
rbits=read_bits(fpath)

filesize=os.path.getsize(fpath)*8
print('filesize:'+str(filesize))


print(rbits)
assert(rbits==inbits)


# str(rbits)

# rbits=rbits[:(t-(32 - lenin)):-1]

     # eight_bits = '00000000'
     # eight_bits[]
     
     
     
    
