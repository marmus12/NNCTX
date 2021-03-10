#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:57:06 2021

@author: root
"""

import numpy as np
from matplotlib import pyplot as plt
# def unique_rows(data):
#     uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
#     return uniq.view(data.dtype).reshape(-1, data.shape[1])


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
    return AB_diff_rows,diff


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
