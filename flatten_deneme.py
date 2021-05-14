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

flatF = np.zeros(2*(z2-z1)*(x2-x1),int)
arr1 = np.array(range(36)).reshape((6,6))
arr2 = np.array(range(36,72)).reshape((6,6))
flatF1 = arr1[z1:z2,x1:x2].flatten('F')
flatF2 = arr2[z1:z2,x1:x2].flatten('F')

flatF[0:(z2-z1)*(x2-x1)] =flatF1
flatF[(z2-z1)*(x2-x1):] =flatF2

flatnew = np.stack((arr1[z1:z2,x1:x2],arr2[z1:z2,x1:x2]),2).flatten('F')




flatold = np.zeros(2*(z2-z1)*(x2-x1),int)
iTin = 0
for xi in range(x1,x2):                
    for zi in range(z1,z2):
        flatold[iTin] = arr1[zi,xi]
        iTin+=1 
for xi in range(x1,x2):                
    for zi in range(z1,z2):
        flatold[iTin] = arr2[zi,xi]
        iTin+=1 
        
assert(np.prod(flatold==flatF))      
        
print(flatold)
print(flatnew)