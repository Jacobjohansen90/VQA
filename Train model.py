#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:27:24 2019

@author: jacob
"""
import json


import Functions as func
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import h5py
import argparse


"""
All network params are set in models.py
"""

#%% Setup Params
#TODO Clean up

parser = argparse.ArgumentParser()

#Datapaths
parser.add_argument('--train_questions_h5', default='../Data/h5py/questions_h5py_train')
parser.add_argument('--train_features_h5', default='../Data/h5py/img_features_h5py_train')
parser.add_argument('--val_questions_h5', default='../Data/h5py/questions_h5py_val')
parser.add_argument('--val_features_h5', default='../Data/h5py/img_features_h5py_val')
parser.add_argument('--vocab_json', default='../Data/vocab/vocab.json') #TODO where should this point

#
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--loader_num_workers', type=int, default=1)

#Samples and shuffeling
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=10000, type=int)
parser.add_argument('--shuffle_train_data', default=True, type=int)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG',
        choices=['PG', 'EE', 'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# LSTM options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument('--cnn_pooling', default='maxpool2',
        choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
        choices=['maxpool2', 'maxpool4', 'none'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0, type=float)

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='data/checkpoint.pt')
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--record_loss_every', type=int, default=1)
parser.add_argument('--checkpoint_every', default=10000, type=int)

#%%Train loop

args = parser.parse_args()
vocab = func.load_vocab(args.vocab_json)

train_loader_kwargs = {
        'question_h5': args.train_questions_h5,
        'feature_h5': args.train_features_h5,
        'vocab':vocab,
        'batch_size':args.batch_size,
        'shuffle': args.shuffle_train_data,
        'max_samples': args.num_train_samples,
        'num_workers': args.loader_num_workers}

val_loader_kwargs = {
        'question_h5': args.val_questions_h5,
        'feature_h5': args.val_features_h5,
        'vocab': vocab,
        'batch_size': args.batch_size,
        'max_samples': args.num_val_samples,
        'num_workers': args.loader_num_workers}

with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
     ClevrDataLoader(**val_loader_kwargs) as val_loader:
    program_generator, pg_kwargs, pg_optimizer = None, None, None
    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    
    pg_best_state, ee_best_state = None, None
    
    #Set up model
    if args.model_type == 'PG' or args.model_type == 'PG+EE':
        program_generator, pg_kwargs = func.get_program_generator(vocab, args)
        pg_optimizer = torch.optim.Adam(program_generator.parameters(), 
                                        lr=args.learning_rate)
        print('Here is the program generator:')
        print(program_generator)
    if args.model_type == 'EE' or args.model_type == 'PG+EE':
        execution_engine, ee_kwargs = func.get_execution_engine(vocab, args)
        ee_optimizer = torch.optim.Adam(execution_engine.parameters(),
                                        lr=args.learning_rate)
        print('Here is the execution engine:')
        print(execution_engine)
        
    #TODO Implement baseline here?
    
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    
    stats = {'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
             'train_accs':[], 'val_accs': [], 'val_accs_ts': [],
             'best_val_acc': -1, 'model_t': 0}
    
    t, epoch, reward_moving_avg = 0,0,0
    
        
        
    