#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:58:36 2021

@author: root
"""
import numpy as np
x1 = 2
x2 = 5
z1 = 1
z2 = 4

arr = np.array(range(36)).reshape((6,6))
flatF = arr[z1:z2,x1:x2].flatten('F')


flatold = np.zeros((z2-z1)*(x2-x1),int)
iTin = 0
for xi in range(x1,x2):                
    for zi in range(z1,z2):
        flatold[iTin] = arr[zi,xi]
        iTin+=1 
        
assert(np.prod(flatold==flatF))      
        
        