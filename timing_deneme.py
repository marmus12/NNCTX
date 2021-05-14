#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:48:48 2021

@author: root
"""


from usefuls import show_time_spent
from time import time
import numpy as np
from array import array

size = 1000
tmax=50000

#%%


start = time()
for t in range(tmax):
    arr2 = np.zeros((size,),'bool')
    for i in range(1,size ):
        arr2[i] = 1
        a = arr2[i-1]

end = time()

show_time_spent(end-start)

#%%
start = time()
for t in range(tmax):
    arr =array('b')
    for i in range(1,size ):
        arr.append(1)
        a = arr[i-1]
end = time()

show_time_spent(end-start)
