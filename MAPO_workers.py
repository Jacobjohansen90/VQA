#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:32:31 2019

@author: jacob
"""

import torch
from torch.autograd import Variable
from DataLoader import ClevrDataLoader
from probables import CountingBloomFilter as CBF
import os
import time
import random

#%% Options


def MAPO(args, pg, ee, vocab, que):
    loader_kwargs = {'question_h5': args.train_questions_h5,
                     'feature_h5': args.train_features_h5,
                     'vocab': vocab,
                     'batch_size': args.batch_size}
    start_t = time.time()
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
                q_name = '-'.join(str(e) for e in question.numpy() if e != 0)
                bf_path = args.bf_load_path + '/' + q_name
                try:
                    bf = CBF(filepath=bf_path)
                except:
                    bf = CBF(est_elements=args.bf_est_ele, 
                             false_positive_rate=args.bf_false_pos_rate)
                feats = feats.unsqueeze(0)
                answer = answer.unsqueeze(0)
                with torch.no_grad():
                    question_var = Variable(question.type(dtype).long())
                    feats_var = Variable(feats.type(dtype))
                program_pred, bf, m_out, m_probs, m_m = pg.reinforce_sample_MAPO(question_var, bf, 
                                                            temperature=args.temperature, 
                                                            argmax=args.MAPO_sample_argmax)
                bf.export(bf_path)
                scores = ee(feats_var, program_pred)
                _, pred = scores.data.cpu().max(1)
                directory = args.high_reward_path+q_name+'/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                if pred == answer:
                    name = str(time.time() - start_t)
                    torch.save(program_pred, directory + name)
                    put_in_que = True
                #Random sample a high reward output
                if len(os.listdir(directory)) == 0:
                    continue
                    #TODO: What to do if no high reward paths?
                else:
                    question = torch.zeros(args.length_output)
                    q = random.choice(os.listdir(directory))
                    program_pred = torch.load(directory + q)
                    q = q.split('-')
                    for i in len(q):
                        question[i] = int(q[i])
                    with torch.no_grad():
                        question_var = Variable(question)
                    hr_m_out, hr_m_probs, hr_m_m, w = pg.random_sample_MAPO(question_var, program_pred,
                                                                   temperature=args.temperature)
                    
                    if args.MAPO_alpha > w:
                        w = args.MAPO_alpha
                    que.put((feats_var, program_pred, answer, hr_m_out, hr_m_probs, hr_m_m, w))
                    if put_in_que:
                        que.put((feats_var, program_pred, answer, m_out, m_probs, m_m, (1-w)))
                        put_in_que = False
                    
                    
                    
                    
                
                
        
        
        
        
        
