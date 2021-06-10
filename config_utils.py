#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:19:16 2021

@author: root
"""
def get_model_info(model_type):
    if model_type=='fast':
        enc_functs_file = 'enc_functs_fast44d'
        log_id = '20210421-180239'
        ctx_type = 100
    elif model_type=='n75':
        enc_functs_file = 'enc_functs_fast44nonext'
        log_id = '20210605-005849' 
        ctx_type = 75
    elif model_type=='n50':
        enc_functs_file = 'enc_functs_fast44_50'
        log_id = '20210606-011442'  
        ctx_type = 50      
    elif model_type=='n36':
        enc_functs_file = 'enc_functs_fast44d'
        log_id ='20210607-161634' 
        ctx_type = 36
    elif model_type=='slow':
        enc_functs_file = 'enc_functs_slow35'
        log_id = '20210409-225535'#'20210415-222905'#
        ctx_type = 100
    return enc_functs_file,log_id,ctx_type