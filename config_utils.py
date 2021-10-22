#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:19:16 2021

@author: root
"""
def get_model_info(model_type,for_train=False):
    if model_type=='fNNOC':
        if for_train:
            enc_functs_file = 'enc_functs_fast44d'
        else:
            enc_functs_file = 'enc_functs_fast45d'
        log_id = '20210421-180239'
        ctx_type = 100
    elif model_type=='fNNOC1':
        enc_functs_file = 'enc_functs_fast44nonext'
        log_id = '20210605-005849' 
        ctx_type = 75
    elif model_type=='fNNOC2':
        enc_functs_file = 'enc_functs_fast44_50'
        log_id = '20210606-011442'  
        ctx_type = 50      
    elif model_type=='fNNOC3':
        enc_functs_file = 'enc_functs_fast45d'
        log_id ='20210607-161634' 
        ctx_type = 36
    elif model_type=='NNOC':
        if for_train:
            enc_functs_file = 'enc_functs_slow35'
        else:
            enc_functs_file = 'enc_functs_slow38'    
        log_id = '20210409-225535'#'20210415-222905'#
        ctx_type = 100
    else:
        raise ValueError("invalid model type")
    return enc_functs_file,log_id,ctx_type
