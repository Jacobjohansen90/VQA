#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:05:48 2019

@author: jacob
"""

import json


import Functions as func
import torch
from torch.autograd import Variable
import argparse
from DataLoader import ClevrDataLoader
import torch.multiprocessing as mp
from MAPO_workers import MAPO

import re
import datetime

import time

parser = argparse.ArgumentParser()

# Start from an existing checkpoint
parser.add_argument('--pg_start_from', default="../Data/models/PG_10k.pt")
parser.add_argument('--ee_start_from', default="../Data/models/EE_10k.pt")
parser.add_argument('--mapo', default=True)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG+EE',
        choices=['PG', 'EE', 'PG+EE'])
parser.add_argument('--train_pg', default=1, type=int)
parser.add_argument('--train_ee', default=1, type=int)
parser.add_argument('--balanced_loss', default=False)

#Training length
parser.add_argument('--num_iterations', default=200000, type=int)
parser.add_argument('--epochs', default=0, type=int) 
#If 0 epochs we use num_iterations to determine training length
parser.add_argument('--break_after', default=10, type=int)
#If val has not improved after break_after checks, we early stop

parser.add_argument('--info', default=False)
#Do you want all info or minimal?

#Samples and shuffeling
parser.add_argument('--num_train_samples', default=None, type=int) #None = All
parser.add_argument('--num_val_samples', default=15000, type=int)
parser.add_argument('--shuffle_train_data', default=False, type=int)

#Bloom Filter
parser.add_argument('--bf_est_ele', default=10**3, type=int)
parser.add_argument('--bf_false_pos_rate', default=0.01, type=float)
parser.add_argument('--bf_load_path', default='../Data/bloom_filters')

parser.add_argument('--bloom_percentage', default=0.05, type=float)

#MAPO
parser.add_argument('--MAPO_use_GPU', default=0, type=int) 
parser.add_argument('--MAPO_qsize', default=320, type=int)
parser.add_argument('--MAPO_sample_argmax', default=True)
parser.add_argument('--MAPO_check_bf', default=0.01, type=float)
parser.add_argument('--MAPO_check_bf_argmax', default=False)
parser.add_argument('--MAPO_split', default=None, type=float)
#How many samples should train PG vs assume PG is correct and only train EE

#Datapaths
parser.add_argument('--train_questions_h5', default='../Data/h5py/questions_h5py_train')
parser.add_argument('--train_features_h5', default='../Data/h5py/img_features_h5py_train')
parser.add_argument('--val_questions_h5', default='../Data/h5py/questions_h5py_val')
parser.add_argument('--val_features_h5', default='../Data/h5py/img_features_h5py_val')
parser.add_argument('--vocab_json', default='../Data/vocab/vocab.json') 
parser.add_argument('--high_reward_path', default='../Data/high_reward_paths/')

parser.add_argument('--checkpoint_path', default='../Data/models/')

#Dataloader params
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--loader_num_workers', type=int, default=0)

# LSTM options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)
parser.add_argument('--length_output', default=30, type=int)

# Module net options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

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
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--reward_decay', default=0.99, type=float)
parser.add_argument('--temperature', default=1.0, type=float)

# Output options
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--record_loss_every', type=int, default=1)
parser.add_argument('--checkpoint_every', default=100, type=int)

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

train_loader = ClevrDataLoader(**train_loader_kwargs)    
