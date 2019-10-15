#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:32:31 2019

@author: jacob
"""

import torch
from torch.autograd import Variable
from DataLoader import ClevrDataLoader


#%% Options


def MAPO(args, pg, ee, vocab, bloom_filter, que):
    loader_kwargs = {'question_h5': args.train_questions_h5,
                     'feature_h5': args.train_features_h5,
                     'vocab': vocab,
                     'batch_size': args.batch_size}
    
    with ClevrDataLoader(**loader_kwargs) as loader:
        dtype = torch.FloatTensor
        if args.MAPO_use_GPU == 1:
            dtype = torch.cuda.FloatTensor
        pg.type(dtype)
        ee.type(dtype)
   
        for batch in loader:
            for i in range(args.batch_size):
                question, _, feats, answer, _, _ = [item[i] for item in batch]
                question = question.unsqueeze(0)
                feats = feats.unsqueeze(0)
                answer = answer.unsqueeze(0)
                with torch.no_grad():
                    question_var = Variable(question.type(dtype).long())
                    feats_var = Variable(feats.type(dtype))
                
                program_pred = pg.reinforce_sample_MAPO(question_var, bloom_filter,
                                                   temperature=args.temperature,
                                                   argmax=args.MAPO_sample_argmax)
                
                scores = ee(feats_var, program_pred)
                _, pred = scores.data.cpu().max(1)
                if pred == answer:
                    #We put these in que instead of scores, as we might
                    #want to make a backwards pass through the execution engine
                    #If this is not the case, you can put scores directly and
                    #save computations
                    que.put((feats_var, program_pred))
                    
                    
                    
                    
                    
                
                
        
        
        
        
        
