#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:30:42 2021

@author: root
"""



from ac_functs import ac_model2
from ac_functs import arithmeticcoding



bsfile = 'bsfile.dat'

encm = ac_model2(nsyms = 2,bsfile=bsfile,ENC=1)


freqslist = [80,10]
freqs = arithmeticcoding.CheckedFrequencyTable(arithmeticcoding.SimpleFrequencyTable(freqslist))

for isy in range(100):
    encm.encode_symbol(freqs,0)
    encm.encode_symbol(freqs,1)
    encm.encode_symbol(freqs,1)
    encm.encode_symbol(freqs,1)
    encm.encode_symbol(freqs,1)
    encm.encode_symbol(freqs,1)
    encm.encode_symbol(freqs,1)
    encm.encode_symbol(freqs,1)
    encm.encode_symbol(freqs,1)
    
encm.end_encoding()
#%%
assert(False)

#%%

from ac_functs import ac_model2
from ac_functs import arithmeticcoding


bsfile = 'bsfile.dat'
freqslist = [80,10]
freqs = arithmeticcoding.CheckedFrequencyTable(arithmeticcoding.SimpleFrequencyTable(freqslist))
decm = ac_model2(nsyms = 2,bsfile=bsfile,ENC=0)

syms =[]
for i in range(40):
    syms.append(decm.decode_symbol(freqs))


decm.end_decoding()











