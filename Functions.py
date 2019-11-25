#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:18:01 2019

@author: jacob
"""

import json
import torch
from LSTM_Model import Seq2Seq
from Module_Model import ModuleNet
from torch.autograd import Variable
from Preprocess_funcs import decode
from DataLoader import ClevrDataset
import re
import h5py
import os
import shutil

#Auto model anmer
def auto_namer(args):
    if args.model_type == "PG":
        if args.num_train_samples is not None:
            model_name = args.model_type+'_'+str(int(args.num_train_samples)//1000)+'k'
        else:
            model_name = args.model_type+'_'+'700k'
    elif args.mapo == True:
        model_name = 'MAPO'+'_'+re.findall(r'[0-9]+',args.pg_start_from)[0]+'k'
    else:
        model_name = args.model_type+'_'+re.findall(r'[0-9]+',args.pg_start_from)[0]+'k'
    return model_name


#Vocab funcs
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
        if cur_vocab_size != len(vocab['question_token_to_idx']):
            print('Expanding vocab size and program generator')
            pg.expand_encoder_vocab(vocab)
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
    if args.info:
        print('Here is the program generator:')
        print(pg)
    return pg, kwargs

#Execution Engine
def load_execution_engine(path, info):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    kwargs = checkpoint['execution_engine_kwargs']
    state = checkpoint['execution_engine_state']
    model = ModuleNet(info, **kwargs)
    model.load_state_dict(state)
    return model, kwargs

def get_execution_engine(vocab, args):
    if args.ee_start_from is not None:
        ee, kwargs = load_execution_engine(args.ee_start_from, args.info)
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
        ee = ModuleNet(args.info, **kwargs)
    ee.cuda()
    ee.train()
    if args.info:
        print('Here is the execution engine:')
        print(ee)
    return ee, kwargs

#Model funcs
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
        with torch.no_grad():
            questions_var = Variable(questions.cuda())
            feats_var = Variable(feats.cuda())
        if programs[0] is not None:
            with torch.no_grad():
                programs_var = Variable(programs.cuda())
        
        scores = None
        if args.model_type == 'PG':
            vocab = load_vocab(args.vocab_json)
            for i in range(questions.size(0)):
                with torch.no_grad():
                    program_pred = program_generator.sample(Variable(questions[i:i+1].cuda()))
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
    acc = round(acc, 4)
    return acc
    
def checkpoint_func(args, program_generator, execution_engine,
                    train_loader, val_loader, t, epoch, stats,
                    model_name, _loss, pg_kwargs, ee_kwargs, 
                    vocab, break_counter):
    if args.info:
        print('Calculating accuracy')
    #train_acc = check_accuracy(args, program_generator,
    #                            execution_engine, train_loader)
    train_acc = 0.5
    val_acc = check_accuracy(args, program_generator,
                              execution_engine, val_loader)
    stats['train_accs'].append(train_acc)
    stats['val_accs'].append(val_acc)
    stats['val_accs_ts'].append(t)
    stats['epoch'].append(epoch)
    if val_acc > stats['best_val_acc']:
        stats['best_val_acc'] = val_acc
        stats['best_model_t'] = t
        best_pg_state = get_state(program_generator)
        best_ee_state = get_state(execution_engine)
        break_counter = 0
        improved_val = "+"
    else: 
        break_counter += 1
        improved_val = "-"
    print('%s - %d - %f \t Train acc: %.4f \t Val acc: %.4f (%s)'  \
          % (model_name, t, sum(_loss)/len(_loss), train_acc, val_acc, improved_val))
    _loss = []
        
    checkpoint = {'args': args.__dict__,
                  'program_generator_kwargs': pg_kwargs,
                  'program_generator_state': best_pg_state,
                  'execution_engine_kwargs': ee_kwargs,
                  'execution_engine_state': best_ee_state,
                  'vocab': vocab}
    for k, v in stats.items():
        checkpoint[k] = v
    if args.info:
        print('Saving checkpoint to %s' % args.checkpoint_path+'.pt')
    torch.save(checkpoint, args.checkpoint_path+'.pt')
    del checkpoint['program_generator_state']
    del checkpoint['execution_engine_state']
    with open(args.checkpoint_path + '.json', 'w') as f:
        json.dump(checkpoint, f)
    return stats, break_counter

#MAPO Functions  
def MAPO_loader(loader_kwargs, loader_que, ee_que, skip_que, max_size):
    if 'question_h5' not in loader_kwargs:
            raise ValueError('Must give question_q5')
    if 'feature_h5'  not in loader_kwargs:
        raise ValueError('Must give feature_h5')
    if 'vocab' not in loader_kwargs:
        raise ValueError('Must give vocab')
        
    feature_h5_path = loader_kwargs.pop('feature_h5')
    feature_h5 = h5py.File(feature_h5_path, 'r')
    
    image_h5 = None
    if 'image_h5' in loader_kwargs:
        image_h5_path = loader_kwargs.pop('image_h5')
        image_h5 = h5py.File(image_h5_path, 'r')
    
    vocab = loader_kwargs.pop('vocab')
    mode = loader_kwargs.pop('mode', 'prefix')
    
    max_samples = loader_kwargs.pop('max_samples', None)
    question_h5_path = loader_kwargs.pop('question_h5')
    image_idx_start_from = loader_kwargs.pop('image_idx_start_from', None)
    
    
    with h5py.File(question_h5_path, 'r') as question_h5:
        dataset = ClevrDataset(question_h5, feature_h5, vocab, mode,
                                        image_h5=image_h5,
                                        max_samples=max_samples,
                                        image_idx_start_from=image_idx_start_from)

        skip_list = []
        max_iterator = len(dataset.all_answers)
        i = 0
        j = 0
        while True:
            if loader_que.qsize() < max_size:
                loader_que.put(dataset[i])
                i += 1
                if i % max_iterator == 0:
                    i = 0
            if ee_que.qsize() < max_size:
                if j in skip_list:
                    j += 1
                    if j % max_iterator == 0:
                        j = 0
                else:
                    sample = dataset[j]
                    q, _, feat, ans, _, _, _ = sample
                    ee_que.put((q, feat, ans))
            for _ in range(skip_que.qsize()):
                index = skip_que.get()
                if index < 0:
                    skip_list.remove(abs(index))
                else:
                    skip_list.append(index)
            

def clean_up(args):
    bf_path = args.bf_load_path
    hr_path = args.high_reward_path 
           
    
    

            