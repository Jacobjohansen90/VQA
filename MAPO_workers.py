#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:32:31 2019

@author: jacob
"""

import torch
from probables import CountingBloomFilter as CBF
import os


#%% MAPO Worker


def MAPO_CPU(args, pg, sample_que, number):
    if args.info:
        print('MAPO process %s started' % str(number))
    dtype = torch.FloatTensor
    pg.type(dtype)
    while True:
        question = sample_que.get()
        q_name = '-'.join(str(e) for e in question if e != 0)
        question = torch.from_numpy(question).long()
        question = question.unsqueeze(0)
        bf_path = args.bf_load_path + q_name
        directory = args.high_reward_path+q_name+'/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            bf = CBF(filepath=bf_path)
        except:
            bf = CBF(est_elements=args.bf_est_ele, 
                     false_positive_rate=args.bf_false_pos_rate)

        for _ in range(args.MAPO_programs_pr_pass):
            if args.multi_GPU and torch.cuda.device_count() > 1:
                program_pred, bf, program_name = pg.module.reinforce_novel_sample(
                        question, bf, temperature=args.temperature, argmax=
                        args.MAPO_sample_argmax)
            else:
                program_pred, bf, program_name = pg.reinforce_novel_sample(question, bf, 
                                                                           temperature=args.temperature, 
                                                                           argmax=args.MAPO_sample_argmax)
            torch.save(program_pred, directory + program_name+'.pt')
        bf.export(bf_path)             
