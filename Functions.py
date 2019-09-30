#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:18:01 2019

@author: jacob
"""

import json
import torch

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

def load_program_generator(path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    model = Seq2Seq(**kwargs)
    model.load_state_dict(state)
    
    return model, kwargs

        
    
def get_program_generator(vocab, pg_start_from):
    if pg_start_from is not None:
        pg, kwargs = load_program_generator(pg_start_from)
        cur_vocab_size = pg.encoder_embed.weight.size(0)
        if cur_vocab_size != len(vocab['question_token_to_idx']):
            print('Expanding vocab size og program generator')
            pg.expand_encoder_vocab(vocab)

