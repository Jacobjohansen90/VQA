#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:52:45 2019

@author: jacob
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import Program_funcs as P
import json
import random

def dataset_to_tensor(dataset, mask=None):
    arr = np.asarray(dataset, dtype=np.int64)
    if mask is not None:
        arr = arr[mask]
    tensor = torch.LongTensor(arr)
    return tensor

class ClevrDataset(Dataset):
    def __init__(self, balanced, oversample, question_h5, feature_h5, vocab, mode='prefix',
                 image_h5=None, max_samples=None, image_idx_start_from=None):
        mode_choices = ['prefix', 'postfix']
        if mode not in mode_choices:
            raise ValueError('Invalid mode "%s"' % mode)
        self.image_h5 = image_h5
        self.vocab = vocab
        self.feature_h5 = feature_h5
        self.mode = mode
        self.max_samples = max_samples
        self.balanced = balanced
        self.oversample = oversample
        
        mask = None
        if image_idx_start_from is not None:
            all_image_idxs = np.asarray(question_h5['image_idxs'])
            mask = all_image_idxs >= image_idx_start_from
            

        self.all_questions = dataset_to_tensor(question_h5['questions'], mask)
        self.all_image_idxs = dataset_to_tensor(question_h5['image_idxs'], mask)
        self.all_programs = None
        if 'programs' in question_h5:
            self.all_programs = dataset_to_tensor(question_h5['programs'], mask)
        self.all_answers = dataset_to_tensor(question_h5['answers'], mask)
        
    def __getitem__(self, index):
        question = self.all_questions[index]
        image_idx = self.all_image_idxs[index]
        answer = self.all_answers[index]
        program_seq = None
        if self.all_programs is not None:
            program_seq = self.all_programs[index]
            
        image = None
        if self.image_h5 is not None:
            image = self.image_h5['images'][image_idx]
            image = torch.FloatTensor(np.asarray(image, dtype=np.float32))
        
        feats = self.feature_h5['features'][image_idx]
        feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))
        
        program_json = None
        if program_seq is not None:
            program_json_seq = []
            for fn_idx in program_seq:
                fn_str = self.vocab['program_idx_to_token'][fn_idx.item()]
                if fn_str == '<START>' or fn_str == '<END>':
                    continue
                fn = P.str_to_function(fn_str)
                program_json_seq.append(fn)
            if self.mode == 'prefix':
                program_json = P.prefix_to_list(program_json_seq)
            elif self.mode == 'postfix':
                program_json = P.postfix_to_list(program_json_seq)
        return (question, image, feats, answer, program_seq, program_json, index)
    
    def __len__(self):
        if self.max_samples is None:
            return self.all_questions.size(0)
        else:
            return min(self.max_samples, self.all_questions.size(0))

class ClevrDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if 'question_h5' not in kwargs:
            raise ValueError('Must give question_q5')
        if 'feature_h5'  not in kwargs:
            raise ValueError('Must give feature_h5')
        if 'vocab' not in kwargs:
            raise ValueError('Must give vocab')
        model = kwargs.pop('model')
        oversample, balanced = None, None
        feature_h5_path = kwargs.pop('feature_h5')
        self.feature_h5 = h5py.File(feature_h5_path, 'r')
        
        self.image_h5 = None
        if 'image_h5' in kwargs:
            image_h5_path = kwargs.pop('image_h5')
            self.image_h5 = h5py.File(image_h5_path, 'r')
        
        if model == 'PG' and 'balanced' in kwargs:
            balanced = kwargs.pop('balanced')
        elif model == 'EE' and 'oversample' in kwargs:
            oversample = kwargs.pop('oversample')
            
        vocab = kwargs.pop('vocab')
        mode = kwargs.pop('mode', 'prefix')
        
        max_samples = kwargs.pop('max_samples', None)
        question_h5_path = kwargs.pop('question_h5')
        image_idx_start_from = kwargs.pop('image_idx_start_from', None)
        with h5py.File(question_h5_path, 'r') as question_h5:
            self.dataset = ClevrDataset(balanced, oversample, question_h5, self.feature_h5, vocab, mode,
                                        image_h5=self.image_h5,
                                        max_samples=max_samples,
                                        image_idx_start_from=image_idx_start_from)
        kwargs['collate_fn'] = self.clevr_collate
        super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)
    
    def close(self):
        if self.image_h5 is not None:
            self.image_h5.close()
        if self.feature_h5 is not None:
            self.feature_h5.close()
    
    def __enter__(self):
        return self
            
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def clevr_collate(self, batch):
        transposed = list(zip(*batch))
        question_batch = default_collate(transposed[0])
        image_batch = transposed[1]
        if any(img is not None for img in image_batch):
            image_batch = default_collate(image_batch)
        feat_batch = transposed[2]
        if any(f is not None for f in feat_batch):
            feat_batch = default_collate(feat_batch)
        answer_batch = default_collate(transposed[3])
        program_seq_batch = transposed[4]
        if transposed[4][0] is not None:
            program_seq_batch = default_collate(transposed[4])
        program_struct_batch = transposed[5]
        i = transposed[6]
        return [question_batch, image_batch, feat_batch, answer_batch, program_seq_batch, program_struct_batch, i]
        
class MyClevrDataLoader():
    def __init__(self, **kwargs):
        if 'question_h5' not in kwargs:
            raise ValueError('Must give question_q5')
        if 'feature_h5'  not in kwargs:
            raise ValueError('Must give feature_h5')
        if 'vocab' not in kwargs:
            raise ValueError('Must give vocab')

        self.oversample, self.balanced = False, False
        self.shuffle = False

        if 'model' in kwargs:
            model = kwargs.pop('model')
            self.shuffle = kwargs.pop('shuffle')

            self.path_to_index = kwargs.pop('path_to_index')
            if model == 'PG' and 'balanced' in kwargs:
                self.balanced = kwargs.pop('balanced')
                self.balanced_n = kwargs.pop('balanced_n')
                if self.balanced_n is not None and self.balanced_n > 24:
                    self.balanced_n = 24
                    print('Number of samples for each answer key is > 24, setting it to 24')
                with open(self.path_to_index) as f:
                    self.index_list = json.load(f)
            elif model == 'EE' and 'oversample' in kwargs:
                self.oversample = kwargs.pop('oversample')
                with open(self.path_to_index) as f:
                    self.index_list = json.load(f)

        feature_h5_path = kwargs.pop('feature_h5')
        self.feature_h5 = h5py.File(feature_h5_path, 'r')
        self.batch_size = kwargs.pop('batch_size')
        self.image_h5 = None
        if 'image_h5' in kwargs:
            image_h5_path = kwargs.pop('image_h5')
            self.image_h5 = h5py.File(image_h5_path, 'r')
                    
        self.vocab = kwargs.pop('vocab')
        self.mode = kwargs.pop('mode', 'prefix')
        
        self.max_samples = kwargs.pop('max_samples', None)
        self.question_h5_path = kwargs.pop('question_h5')
        self.image_idx_start_from = kwargs.pop('image_idx_start_from', None)
        
        mode_choices = ['prefix', 'postfix']
        if self.mode not in mode_choices:
            raise ValueError('Invalid mode "%s"' % self.mode)
        
        mask = None
        if self.image_idx_start_from is not None:
            all_image_idxs = np.asarray(self.question_h5['image_idxs'])
            mask = all_image_idxs >= self.image_idx_start_from

        self.question_h5 = h5py.File(self.question_h5_path, 'r')
        
        self.all_questions = dataset_to_tensor(self.question_h5['questions'], mask)
        self.all_image_idxs = dataset_to_tensor(self.question_h5['image_idxs'], mask)
        self.all_programs = None
        if 'programs' in self.question_h5:
            self.all_programs = dataset_to_tensor(self.question_h5['programs'], mask)
        self.all_answers = dataset_to_tensor(self.question_h5['answers'], mask)
        
        self.index = 0
        self.mode = 'Train'
        
        if self.balanced and model == 'PG':
            self.sample_list = []
            for i in range(28):
                indexs = random.sample(self.index_list[str(i+4)], self.balanced_n)
                for index in indexs:
                    self.sample_list.append(index)
        elif self.oversample and model == 'EE':
            max_n = 0
            for i in range(28):
                if len(self.index_list[str(i+4)]) > max_n:
                    max_n = len(self.index_list[str(i+4)])
            self.sample_list = list(range(max_n*28))
            for i in range(28):
                current_list = self.index_list[str(i+4)]
                max_len = len(current_list)
                k = 0
                for j in range(max_n):
                    self.sample_list[(i*max_n)+j] = current_list[k]
                    k += 1
                    if k % max_len == 0:
                        k = 0           
                          
        else:
            if self.max_samples is not None:
                self.sample_list = random.sample(range(len(self.all_questions)), self.max_samples)
            else:
                self.sample_list = list(range(len(self.all_questions)))
        if self.shuffle:
            self.shuffle_samples()
            
        self.max_index = len(self.sample_list)
        
        self.questions = torch.zeros(self.batch_size,46).long()
        self.feats = torch.zeros(self.batch_size,1024,14,14) 
        self.answers = torch.zeros(self.batch_size,1).long()
        self.indexs = torch.zeros(self.batch_size,1).long()
        self.program_seq = torch.zeros(self.batch_size,30).long()
        self.images = None #torch.zeros(self.batch_size)
        self.program_struct = []
        
    def batch(self):
        if self.mode == 'Train':
            for j in range(self.batch_size):
                index = self.sample_list[self.index]
                self.index += 1
                
                question = self.all_questions[index]
                image_idx = self.all_image_idxs[index]
                answer = self.all_answers[index]
                program_seq = None
                
                if self.all_programs is not None:
                    program_seq = self.all_programs[index]
                    
                image = None
                if self.image_h5 is not None:
                    image = self.image_h5['images'][image_idx]
                    image = torch.FloatTensor(np.asarray(image, dtype=np.float32))
                
                feats = self.feature_h5['features'][image_idx]
                feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))
                
                program_json = None
                if program_seq is not None:
                    program_json_seq = []
                    for fn_idx in program_seq:
                        fn_str = self.vocab['program_idx_to_token'][fn_idx.item()]
                        if fn_str == '<START>' or fn_str == '<END>':
                            continue
                        fn = P.str_to_function(fn_str)
                        program_json_seq.append(fn)
                    if self.mode == 'prefix':
                        program_json = P.prefix_to_list(program_json_seq)
                    elif self.mode == 'postfix':
                        program_json = P.postfix_to_list(program_json_seq)
                
                self.questions[j] = question
                self.feats[j] = feats 
                self.answers[j] = answer
                self.indexs[j] = index
                self.program_seq[j][0:len(program_seq)] = program_seq
                self.program_struct.append(program_json)
                
                if self.index == self.max_index:
                    self.index = 0
                    self.shuffle_samples()
                   
            return [self.questions, self.images, self.feats, self.answers, self.program_seq, self.program_struct, self.indexs]
        elif self.mode == 'eval':
            for j in range(self.batch_size):
                index = self.sample_list[self.eval_index]
                self.eval_index += 1
                
                question = self.all_questions[index]
                image_idx = self.all_image_idxs[index]
                answer = self.all_answers[index]
                program_seq = None
                
                if self.all_programs is not None:
                    program_seq = self.all_programs[index]
                    
                image = None
                if self.image_h5 is not None:
                    image = self.image_h5['images'][image_idx]
                    image = torch.FloatTensor(np.asarray(image, dtype=np.float32))
                
                feats = self.feature_h5['features'][image_idx]
                feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))
                
                program_json = None
                if program_seq is not None:
                    program_json_seq = []
                    for fn_idx in program_seq:
                        fn_str = self.vocab['program_idx_to_token'][fn_idx.item()]
                        if fn_str == '<START>' or fn_str == '<END>':
                            continue
                        fn = P.str_to_function(fn_str)
                        program_json_seq.append(fn)
                    if self.mode == 'prefix':
                        program_json = P.prefix_to_list(program_json_seq)
                    elif self.mode == 'postfix':
                        program_json = P.postfix_to_list(program_json_seq)
                
                self.questions[j] = question
                self.feats[j] = feats 
                self.answers[j] = answer
                self.indexs[j] = index
                self.program_seq[j][0:len(program_seq)] = program_seq
                self.program_struct.append(program_json)
            
                if self.eval_index == self.max_index:
                    self.mode = 'Train'
                    return [self.questions[:j+1], self.images[:j+1], self.feats[:j+1],
                            self.answers[:j+1], self.program_seq[:j+1], 
                            self.program_struct[:j+1], self.indexs[:j+1], True]
                
            return [self.questions, self.images, self.feats, self.answers, 
                    self.program_seq, self.program_struct, self.indexs, False]
        
    def eval_mode(self):
        self.mode = 'eval'
        self.eval_index = 0
    
    def shuffle_samples(self):
        np.random.shuffle(self.sample_list)
        
