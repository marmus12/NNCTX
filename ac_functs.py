#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:44:56 2021

@author: root
"""


import sys


sys.path.append('/home/emre/Documents/kodlar/Reference-arithmetic-coding-master/python/')

# import contextlib
import arithmeticcoding



class ac_model:
    
     def __init__(self,nsyms,bsfile,ENC):
         self.ENC = ENC
         self.bsfile = bsfile
         

         self.nsyms = nsyms
         self.initfreqs = arithmeticcoding.FlatFrequencyTable(self.nsyms)
         self.freqs = arithmeticcoding.SimpleFrequencyTable(self.initfreqs)
         
         if ENC:
             self.ofp = open(self.bsfile, "wb")
             bitout = arithmeticcoding.BitOutputStream(self.ofp)
             self.enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
         else:
             self.ofp = open(self.bsfile, "rb")
             bitin = arithmeticcoding.BitInputStream(self.ofp)
             # bitin = contextlib.closing(arithmeticcoding.BitOutputStream(self.ofp))              
             self.dec = arithmeticcoding.ArithmeticDecoder(32, bitin)

     def encode_symbol(self,symbol):
        self.enc.write(self.freqs, symbol[0])
        self.freqs.increment(symbol[0])

     def end_encoding(self,):
        self.enc.write(self.freqs, self.nsyms-1)
        self.enc.finish()
        self.ofp.close()


     def decode_symbol(self,):
         symbol = self.dec.read(self.freqs)
         self.freqs.increment(symbol)   
         return symbol

     def end_decoding(self,):
         self.ofp.close()


class ac_model2:
    
     def __init__(self,nsyms,bsfile,ENC):
         self.ENC = ENC
         self.bsfile = bsfile
         

         self.nsyms = nsyms
         # self.initfreqs = arithmeticcoding.FlatFrequencyTable(self.nsyms)
         # self.freqs = arithmeticcoding.SimpleFrequencyTable(self.initfreqs)
         
         if ENC:
             self.ofp = open(self.bsfile, "wb")
             bitout = arithmeticcoding.BitOutputStream(self.ofp)
             self.enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
         else:
             self.ofp = open(self.bsfile, "rb")
             bitin = arithmeticcoding.BitInputStream(self.ofp)
             # bitin = contextlib.closing(arithmeticcoding.BitOutputStream(self.ofp))              
             self.dec = arithmeticcoding.ArithmeticDecoder(32, bitin)

     def encode_symbol(self,freqs,symbol):
        self.enc.update(freqs, symbol)
        # self.freqs.increment(symbol[0])

     def end_encoding(self,):
        # self.enc.write(self.freqs, self.nsyms-1)
        self.enc.finish()
        self.ofp.close()


     def decode_symbol(self,freqs):
         symbol = self.dec.read(freqs)
         # self.freqs.increment(symbol)   
         return symbol

     def end_decoding(self,):
         self.ofp.close()






def compress(inp, bitout):
	initfreqs = arithmeticcoding.FlatFrequencyTable(257)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
	while True:
		# Read and encode one byte
		symbol = inp.read(1)
		if len(symbol) == 0:
			break
		enc.write(freqs, symbol[0])
		freqs.increment(symbol[0])
	enc.write(freqs, 256)  # EOF
	enc.finish()  # Flush remaining code bits
    
def decompress(bitin, out):
	initfreqs = arithmeticcoding.FlatFrequencyTable(257)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
	while True:
		# Decode and write one byte
		symbol = dec.read(freqs)
		if symbol == 256:  # EOF symbol
			break
		out.write(bytes((symbol,)))
		freqs.increment(symbol)   