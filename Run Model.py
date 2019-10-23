#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:32:31 2019

@author: jacob
"""

import argparse

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import h5py

import Functions as func
from DataLoader import ClevrDataLoader

parser = argparse.ArgumentParser()
#%% Settings

#Save predictions?
parser.add_argument('--output_h5', default='../Data/test_700k_pred')

#Path to trained models
parser.add_argument('--program_generator', default='../Data/models/pg_700k_checkpoint.pt')
parser.add_argument('--execution_engine', default='../Data/models/ee_700k_checkpoint.pt')
parser.add_argument('--use_gpu', default=1, type=int)

#Test set path
parser.add_argument('--input_question_h5', default='../Data/h5py/questions_h5py_test')
parser.add_argument('--input_features_h5', default='../Data/h5py/img_features_h5py_test')
parser.add_argument('--use_gt_programs', default=0, type=int)

#If we need a different vocab than the one stored in checkpoint
parser.add_argument('--vocab_json', default=None)

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--sample_argmax', default=1, type=int)
parser.add_argument('--temperature', default=1.0, type=float)

args = parser.parse_args()
#%%
if args.program_generator is not None and args.execution_engine is not None:
    print('Loading program generator from: ', args.program_generator)
    program_generator, _ = func.load_program_generator(args.program_generator)
    print('Loading execution engine from :', args.execution_engine)
    execution_engine, _ = func.load_execution_engine(args.execution_engine,
                                                     verbose=False)
else:
    raise KeyError('Must give path to program generator and execution engine')

if args.vocab_jjson is not None:
    new_vocab = func.load_vocab(args.vocab_json)
    program_generator.expand_encoder_vocab(new_vocab['question_token_to_idx'])


vocab = func.load_vocab(args)
loader_kwargs = {'question_h5': args.input_question_h5,
                 'feature_h5': args.input_features_h5,
                 'vocab': vocab,
                 'batch_size': args.batch_size}

if args.num_samples is not None and args.num_samples > 0:
    loader_kwargs['max_samples'] = args.num_samples
    
with ClevrDataLoader(**loader_kwargs) as loader:
    if args.use_gpu == 1:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    
    program_generator.type(dtype)
    program_generator.eval()
    execution_engine.type(dtype)
    execution_engine.eval()
    
    all_scores, all_programs = [], []
    all_probs = []
    num_correct, num_samples = 0,0
    for batch in loader:
        questions, images, feats, answers, programs, program_lists = batch
        
        with torch.no_grad():
            questions_var = Variable(questions.type(dtype).long())
            feats_var = Variable(feats.type(dtype))
        
        programs_pred = program_generator.reinforce_sample(questions_var,
                                                           temperature=args.temperature,
                                                           argmax=args.sample_argmax)
        if args.use_gt_program == 1:
            scores = execution_engine(feats_var, program_lists)
        else:
            scores = execution_engine(feats_var, programs_pred)
        probs = F.softmax(scores)
        _, preds = scores.data.cpu().max(1)
        all_programs.append(programs_pred.data.cpu().clone())
        all_scores.append(scores.data.cpu().clone())
        all_probs.append(probs.data.cpu().clone())
        
        num_correct += (preds==answers).sum()
        num_samples += preds.size(0)
    
    acc = float(num_correct) / num_samples
    print('Got %d / %d = %.2f correct' & (num_correct, num_samples, 100*acc))
    
    all_scores = torch.cat(all_scores, 0)
    all_probs = torch.cat(all_probs, 0)
    all_programs = torch.cat(all_programs, 0)
    if args.output_h5 is not None:
        print('Writing output to "%s"' % args.output_h5)
        with h5py.File(args.output_h5, 'w') as file:
            file.create_dataset('scores', data=all_scores.numpy())
            file.create_dataset('probs', data=all_probs.numpy())
            file.create_dataset('predicted_programs', data=all_programs.numpy())
