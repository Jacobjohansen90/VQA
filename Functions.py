#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:18:01 2019

@author: jacob
"""

import json
import torch
from Models import Seq2Seq, 

def invert_dict(d):
    return {v: k for k, v in d.items()}

def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        #TODO implement program if needed
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    
    #Make sure tokens are correct
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    
    return vocab

def parse_int_list(s):
    return tuple(int(n) for n in s.split(','))

#Program Generator
def load_program_generator(path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    model = Seq2Seq(**kwargs)
    model.load_state_dict(state)
    
    return model, kwargs
          
def get_program_generator(vocab, args):
    if args.pg_start_from is not None:
        pg, kwargs = load_program_generator(args.pg_start_from)
        cur_vocab_size = pg.encoder_embed.weight.size(0)
        if cur_vocab_size != len(args.vocab['question_token_to_idx']):
            print('Expanding vocab size og program generator')
            pg.expand_encoder_vocab(args.vocab)

#Execution Engine
def load_execuion_engine(path, verbose=True):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    model = Seq2Seq(**kwargs)
    model.load_state_dict(state)
    return model, kwargs

def get_execution_engine(vocab, args):
    if args.exec_start_from is not None:
        ee, kwargs = load_execution_engine(args.exec_start_from)
    else:
        kwargs = {'vocab':vocab,
                  'feature_dim': parse_int_list(args.feature_dim),
                  'stem_batchnorm': args.module_stem_batchnorm,
                  'stem_num_layers': args.module_stem_num_layers,
                  'module_dim': args.module_dim,
                  'module_residual': args.module_residual,
                  'module_batchnorm': args.module_batchnorm,
                  'classifier_proj_dim': args.classifier_proj_dim,
                  'classifier_downsample': args.classifier_downsample,
                  'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
                  'classifier_batchnorm': args.classifier_batchnorm,
                  'classifier_dropout': args.classifier_dropout}
        ee = ModuleNet(**kwargs)
    ee.cuda()
    ee.train()
    return ee, kwargs