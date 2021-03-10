#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:12:44 2021

@author: root
"""
import numpy as np
from models import MyModel3
from scipy.io import loadmat,savemat
from usefuls import dests2probs











m=MyModel3()

checkpoint_path = '/home/emre/Documents/train_logs/20210209-134400cp.ckpt'
output_path = 'probs.mat'
# Loads the weights
m.model.load_weights(checkpoint_path)



# val_ctx_path = '/home/emre/Documents/DATA/TempT22_DesT22_soldier_0690.mat'
# val_ctx_dict = loadmat(val_ctx_path)
# val_ctxs = val_ctx_dict['TempT22']
# val_dests = val_ctx_dict['DesT22']



# test_ctxs= val_ctxs[1000:1010]
# test_dests= val_dests[1000:1010]
test_ctx = np.ones(shape=(1,22))
output=m.model(test_ctx).numpy()
savemat(output_path,{'probs':output})
print('output was written to: '+output_path)


#probs = dests2probs(val_ctxs ,val_dests)

#gt_probs=probs[1000:1010]










