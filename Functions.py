#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:18:01 2019

@author: jacob
"""

import json
import torch
from LSTM_Model import Seq2Seq
from torch.autograd import Variable
from Preprocess_funcs import decode

def invert_dict(d):
    return {v: k for k, v in d.items()}

def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    
    #Make sure tokens are correct
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    
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
            kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
    else:
        kwargs = {'encoder_vocab_size': len(vocab['question_token_to_idx']),
                  'decoder_vocab_size': len(vocab['program_token_to_idx']),
                  'wordvec_dim': args.rnn_wordvec_dim,
                  'hidden_dim': args.rnn_hidden_dim,
                  'rnn_num_layers': args.rnn_num_layers,
                  'rnn_dropout': args.rnn_dropout}
        pg = Seq2Seq(**kwargs)
    pg.cuda()
    pg.train()
    return pg, kwargs

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

def set_mode(mode, models):
    assert mode in ['train', 'eval']
    for m in models:
        if m is None:
            continue
        if mode == 'train':
            m.train()
        if mode == 'eval':
            m.eval()

def get_state(m):
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state
            
def check_accuracy(args, program_generator, execution_engine, loader):
    set_mode('eval', [program_generator, execution_engine])
    num_correct, num_samples = 0,0
    for batch in loader:
        questions, _, feats, answers, programs, _ = batch
        
        questions_var = Variable(questions.cuda(), volatile=True)
        feats_var = Variable(feats.cuda(), volatile=True)
        answers_var = Variable(answers.cuda(), volatile=True)
        if programs[0] is not None:
            programs_var = Variable(programs.cuda(), volatile=True)
        
        scores = None
        if args.model_type == 'PG':
            vocab = load_vocab(args.vocab_json)
            for i in range(questions.size(0)):
                program_pred = program_generator.sample(Variable(questions[i:i+1].cuda(),
                                                                 volatile=True))
                program_pred_str = decode(program_pred, vocab['program_idx_to_token'])
                program_str = decode(programs[i], vocab['program_idx_to_token'])
                if program_pred_str == program_str:
                    num_correct += 1
                num_samples += 1

        elif args.model_type == 'EE':
            scores = execution_engine(feats_var, programs_var)

        elif args.model_type == 'PG+EE':
            program_pred = program_generator.reinforce_sample(questions_var,
                                                              argmax=True)
            scores = execution_engine(feats_var, program_pred)
        
        if scores is not None:
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == answers).sum()
            num_samples += preds.size(0)
            
        if num_samples >= args.num_val_samples:
            break
        
    set_mode('train', [program_generator, execution_engine])
    acc = float(num_correct) / num_samples
    return acc
            
            