#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:57:06 2021

@author: root
"""

import numpy as np
from matplotlib import pyplot as plt
from array import array
from dec2bin import dec2bin
# def unique_rows(data):
#     uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
#     return uniq.view(data.dtype).reshape(-1, data.shape[1])

import os

def get_dir_size(directory):
    """Returns the `directory` size in bytes."""
    total = 0
    try:
        # print("[+] Getting the size of", directory)
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                total += get_dir_size(entry.path)
    except NotADirectoryError:
        # if `directory` isn't a directory, get the file size then
        return os.path.getsize(directory)*8
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total*8

def write_bits(inbits,fpath):
    lenin = len(inbits)
    nbytes = np.ceil(lenin/8).astype(int)
    
    
    bin_array = array("B")
    bits = inbits + "0" * (nbytes*8 - lenin)  # Align bits to 32, i.e. add "0" to tail
    for index in range(0, nbytes*8, 8):
        byte = bits[index:index + 8][::-1]
        bin_array.append(int(byte, 2))
    
    with open(fpath, "wb") as f:
        f.write(bytes(bin_array))
#######################################
def read_bits(fpath) :
    with open(fpath, "rb") as f:
        rbyte_array = f.read()
      
    rbits=''
    # t=100-1
    for rbyte in rbyte_array[0:(len(rbyte_array)-1)]:
         red_bits = dec2bin(rbyte)
         nbits = len(red_bits)
         red_bits='0'*(8-nbits)+red_bits
         
         for irb in range(8):
             rbits = rbits+str(red_bits[7-irb])
             # t-=1
    rbyte = rbyte_array[-1]
    red_bits = dec2bin(rbyte)
    for irb in range(len(red_bits)):
        rbits = rbits+str(red_bits[(len(red_bits)-1)-irb]) 
        
    return rbits

def ints2bs(ints,nintbits):
    
    inbits = ''
    for i,inte in enumerate(ints):
        intbin = dec2bin(inte)
        
        inbits=inbits+'0'*(nintbits[i]-len(intbin)) +intbin
        
    return inbits

def write_ints(ints,nintbits,fpath):
    
    inbits = ints2bs(ints,nintbits)
    
    inbits = inbits+'1'

    write_bits(inbits,fpath)

def read_ints(nintbits,fpath):
    
    bits = read_bits(fpath)
    
    ints = bs2ints(bits,nintbits)
    return ints
            
def bs2ints(bits,nintbits):
    ints = np.zeros(shape=(len(nintbits),),dtype=int)
    t=0
    for ii,nintbit in enumerate(nintbits):
        intbits = bits[t:(t+nintbit)]
        theint=0
        for k,intbit in enumerate(intbits):
            theint+=2**(nintbit-1-k)*int(intbit)
        ints[ii] = theint
        t+=nintbit


    return ints        


def plt_imshow(im,figsize=(12,12)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
   #  plt.tight_layout()
    plt.show()

def plt_savepng(im,figsize=(12,12)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    
    fig.savefig('im.png')
    #plt.show()



def find_diff_rows_with_ind(A,B): 
    #use only with unique row A,B matrices
    nA,wA = A.shape
    nB,wB = B.shape
    assert(wA==wB)
    AB_diff_rows = np.zeros_like(A)
    diff_row_indsA = -1*np.ones((nA,))
    n_diff_rows = 0
    for iA in range(nA):
        curr_row = A[iA,:]
        contained=0
        for iB in range(nB):
            if np.all(curr_row==B[iB,:]): #curr. row is contained in B
                contained=1
                break
        if not(contained):
            AB_diff_rows[n_diff_rows,:] = curr_row
            diff_row_indsA[n_diff_rows] = iA
            n_diff_rows+=1
            
    AB_diff_rows = AB_diff_rows[0:n_diff_rows,:]
    diff_row_indsA = diff_row_indsA[0:n_diff_rows]
    return AB_diff_rows,diff_row_indsA


def find_diff_rows(A,B):
    ##A'da olup Bde olmayan Rowlari donduruyor
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}
    
    C = np.setdiff1d(A.view(dtype), B.view(dtype))
    
    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C


def find_common_rows(A,B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}
    
    C = np.intersect1d(A.view(dtype), B.view(dtype))
    
    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C



def setdiff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))
 
    
def dests2probs(temps,dests):
    
    
    uval_ctxs,valinds,valinvs,uval_counts  = np.unique(temps , axis=0,return_index=True,return_inverse=True,return_counts=True)
    
    nu_ctxs = uval_ctxs.shape[0]
    val_01_counts = np.zeros([nu_ctxs,2])
    for ictx in range(nu_ctxs):
        ctx_dests = dests[valinvs==ictx]
        val_01_counts[ictx,1] = np.sum(ctx_dests)
        val_01_counts[ictx,0] = ctx_dests.shape[0]-val_01_counts[ictx,1]
        
    uval_probs = val_01_counts/np.sum(val_01_counts,1,keepdims=True)
    
    probs = uval_probs[valinvs,:]
    return probs



def asvoid(arr):
    """
    View the array as dtype np.void (bytes)
    This views the last axis of ND-arrays as bytes so you can perform comparisons on
    the entire row.
    http://stackoverflow.com/a/16840350/190597 (Jaime, 2013-05)
    Warning: When using asvoid for comparison, note that float zeros may compare UNEQUALLY
    >>> asvoid([-0.]) == asvoid([0.])
    array([False], dtype=bool)
    """
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


def in1d_index(a, b):
    voida, voidb = map(asvoid, (a, b))
    return np.where(np.in1d(voidb, voida))[0]    


def compare_Locations(Loc1,GT):

    uL1 = np.unique(Loc1,axis=0)
    nL1 = uL1.shape[0]
    uGT = np.unique(GT,axis=0)
    nuGT = uGT.shape[0]
    indsG = in1d_index(uL1,uGT)
    indsL1 = in1d_index(uGT,uL1)
    fp_inds = setdiff(list(range(nL1)),indsL1)
    fn_inds = setdiff(list(range(nuGT)),indsG)
    nTP = len(indsG)
    nFN = nuGT - nTP 
    TP = uGT[indsG,:]
    FN = uGT[fn_inds,:]    
    nFP = nL1-nTP
      
    FP = uL1[fp_inds,:]
    
    print('nTP:'+ str(nTP) + ' nFP:'+ str(nFP) + ' nFN:'+ str(nFN) )
    
    
    return TP,FP,FN
    
    
    
def dec2bin2(inte,nbits):
    
    intbin = dec2bin(inte)
    n0s = nbits-len(intbin)
    assert(n0s>=0)
    return '0'*n0s +intbin
    
    
def bin2dec2(bits):
    nbits = len(bits)
    dec = 0
    for ib,bit in enumerate(bits):        
        dec+=2**(nbits-ib-1)*int(bit)
    return dec
    

def show_time_spent(time_spent):
    nmins = int(time_spent//60)
    nsecs = int(np.round(time_spent-nmins*60))
    print('time spent: ' + str(nmins) + 'm ' + str(nsecs) + 's')
    
# a = np.array([[4, 6],[2, 6],[5, 2]])
# b = np.array([[1, 7],[1, 8],[2, 6],[2, 1],[2, 4],[4, 6],[4, 7],[5, 9],[5, 2],[5, 1]])

# print(in1d_index(a, b))