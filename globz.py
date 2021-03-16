#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:54:09 2021

@author: root
"""


import numpy as np
#

def init():
    global esymbs,symbs,isymb
    symbs = -1*np.ones((1000000,),'int')
    isymb = 0