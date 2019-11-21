#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:32:31 2019

@author: jacob
"""

import torch
from torch.autograd import Variable
from probables import CountingBloomFilter as CBF
import os
import random
import time

#%% Options


def MAPO(args, pg, ee, loader_que, vocab, que, skip_que, number):
    if args.info:
        print('MAPO process %s started' % str(number))
    
    dtype = torch.FloatTensor
    if args.MAPO_use_GPU == 1:
        dtype = torch.cuda.FloatTensor
    pg.type(dtype)
    ee.type(dtype)
    while True:
        sample = loader_que.get()
        question, _, feats, answer, _, _, i = sample
        q_name = '-'.join(str(e) for e in question.numpy() if e != 0)
        question = question.unsqueeze(0)
        bf_path = args.bf_load_path + '/' + q_name
        directory = args.high_reward_path+q_name+'/'
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
            
        if random.uniform(0,1) < args.MAPO_check_bf:
            if args.info:
                print('Checking bloom filter is relevant')
            program_pred = pg.reinforce_sample(question_var, argmax=args.MAPO_check_bf_argmax)
            program_name = '-'.join(str(e) for e in program_pred.tolist())
            if bf.check(program_name):
                scores = ee(feats_var, program_pred)
                _, pred = scores.data.cpu().max(1)
                if pred == answer:
                    if program_name not in os.listdir(directory): 
                        os.remove(bf_path)
                        bf = CBF(est_elements=args.bf_est_ele, 
                                 false_positive_rate=args.bf_false_pos_rate)
                else:
                    if program_name+'.pt' in os.listdir(directory):
                        os.remove(directory+program_name+'.pt')
                        for program in os.listdir(directory):
                            program_pred = torch.load(directory+program)
                            scores = ee(feats_var, program_pred)
                            _, pred = scores.data.cpu().max(1)
                            if pred != answer:
                                os.remove(directory+program)
                        if len(os.listdir(directory)) == 0:
                            skip_que.put(-i)
                        os.remove(bf_path)
                        bf = CBF(est_elements=args.bf_est_ele, 
                                 false_positive_rate=args.bf_false_pos_rate)
        program_pred, bf, program_name = pg.reinforce_novel_sample(question_var, bf, 
                                                     temperature=args.temperature, 
                                                     argmax=args.MAPO_sample_argmax)
        bf.export(bf_path)     
        scores = ee(feats_var, program_pred)
        _, pred = scores.data.cpu().max(1)                
        if pred == answer:
            if not os.path.exists(directory):
                os.makedirs(directory)
                skip_que.put(i)           
            torch.save(program_pred, directory + program_name+'.pt')
            #Current prediction is high reward path
            put(args, que, question_var, feats_var, program_pred, answer)
            #Sample unseen (most likely non reward) path
            bf = CBF(filepath=bf_path)
            program_pred = pg.sample_non_high_reward(question_var, 
                                                     temperature=args.temperature, 
                                                     argmax=args.MAPO_sample_argmax)
            put(args, que, question_var, feats_var, program_pred, answer)

        else:
            if not os.path.exists(directory):
                #No high reward exists, soo we skip sample. It will be trained by the EE que
                continue
            else:
                #Our current predictions is non reward path
                put(args, que, question_var, feats_var, program_pred, answer)
                #Sample high reward path
                q = random.choice(os.listdir(directory))
                program_pred = torch.load(directory + q)
                put(args, que, question_var, feats_var, program_pred, answer)

def put(args, que, question, feats, program_pred, ans):
    if que.qsize() >= args.MAPO_qsize:
        time.sleep(1)
        put(args, que, question, feats, program_pred, ans)
    else:
        que.put((question, feats, program_pred, ans))