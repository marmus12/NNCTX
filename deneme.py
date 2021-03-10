#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:28:48 2021

@author: root
"""



from dataset import ctx_dataset


ctx_type=122
train_data_dir = ['/home/emre/Documents/DATA/andrew_david_sarah_6_122/', 
                  '/home/emre/Documents/DATA/longdress_18_122/']


ds = ctx_dataset(train_data_dir,ctx_type)