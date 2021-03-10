#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 22:43:25 2021

@author: root
"""
#from tensorflow.keras.backend import log, mean
from tensorflow.keras import backend as kb
import numpy as np

def CL_criterion(y_true,y_pred):
    
    CL=kb.sum(-y_true[:,0]*kb.log(y_pred[:,0])/kb.log(2.)-y_true[:,1]*kb.log(y_pred[:,1])/kb.log(2.))

    return CL
    
    
def np_CL_criterion(y_true,y_pred):
    
    CL=np.sum(-y_true[:,0]*np.log(y_pred[:,0])/np.log(2.)-y_true[:,1]*np.log(y_pred[:,1])/np.log(2.))

    return CL    
    