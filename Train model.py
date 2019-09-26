#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:27:24 2019

@author: jacob
"""

import Functions as func

h5_train_questions = '../Data/h5py/h5py_train' 
h5_train_features = 
h5_val_questions = '../Data/h5py/h5py_val
h5_val_features = 
json_vocab = '../Data/vocab/vocab.json'

load = True #Load previous model


##Model params
batch_size = 64
shuffle = True
num_train_samples =
num_val_samples =  
num_workers = 

#Layer params
wordvec_dim = 300
lstm_hidden_dim = 256
lstm_layers = 2
lstm_dropout = 0

NMN_layers = 2
NMN_batchnorm = True
NMN_dim = 128
NMN_

CNN_


##Train loop

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
    
    
