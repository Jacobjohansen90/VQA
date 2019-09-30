#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:27:24 2019

@author: jacob
"""

import Functions as func

#%% Datapaths 

h5_train_questions = '../Data/h5py/questions_h5py_train' 
h5_train_features = '../Data/h5py/img_features_h5py_train'
h5_val_questions = '../Data/h5py/questions_h5py_val
h5_val_features = '../Data/h5py/img_features_h5py_val'
json_vocab = '../Data/vocab/vocab.json'

#%% Load options

load = True #Load previous model
checkpoint_path = 1 #Automate pick latest

#%% Model params

batch_size = 32
shuffle = True
num_train_samples = None
num_val_samples = 10000  
num_workers = 
shuffle_train_data = True
model_type = 'PG'

"""
All network params are set in models.py
"""

#%%Train loop

vocab = func.load_vocab(json_vocab)

train_loader_kwargs = {
        'question_h5': h5_train_questions,
        'feature_h5': h5_train_features,
        'vocab':vocab,
        'batch_size':batch_size,
        'shuffle' = shuffle,
        'max_samples': num_train_samples,
        'num_workers': num_workers}

  val_loader_kwargs = {
    'question_h5': h5_val_questions,
    'feature_h5': h5_val_features,
    'vocab': vocab,
    'batch_size': batch_size,
    'max_samples': num_val_samples,
    'num_workers': num_workers}

with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
     ClevrDataLoader(**val_loader_kwargs) as val_loader:
    program_generator, pg_kwargs, pg_optimizer = None, None, None
    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    
    pg_best_state, ee_best_state = None, None
    
    #Set up model
    if model_type == 'PG' or model_type = 'PG+EE':
        program_generator, pg_kwargs = 
    
    
