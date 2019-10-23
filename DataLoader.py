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

def dataset_to_tensor(dataset, mask=None):
    arr = np.asarray(dataset, dtype=np.int64)
    if mask is not None:
        arr = arr[mask]
    tensor = torch.LongTensor(arr)
    return tensor

class ClevrDataset(Dataset):
    def __init__(self, question_h5, feature_h5, vocab, mode='prefix',
                 image_h5=None, max_samples=None, image_idx_start_from=None):
        mode_choices = ['prefix', 'postfix']
        if mode not in mode_choices:
            raise ValueError('Invalid mode "%s"' % mode)
        self.image_h5 = image_h5
        self.vocab = vocab
        self.feature_h5 = feature_h5
        self.mode = mode
        self.max_samples = max_samples
        
        mask = None
        if image_idx_start_from is not None:
            all_image_idxs = np.asarray(question_h5['image_idxs'])
            mask = all_image_idxs >= image_idx_start_from
            
        #We read all data into memory, change this if Q file becomes too large
        print('Reading data into memory')
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
        return (question, image, feats, answer, program_seq, program_json)
    
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
            
        feature_h5_path = kwargs.pop('feature_h5')
        print('Reading features from ', feature_h5_path)
        self.feature_h5 = h5py.File(feature_h5_path, 'r')
        
        self.image_h5 = None
        if 'image_h5' in kwargs:
            image_h5_path = kwargs.pop('image_h5')
            print('Reading images from ', image_h5_path)
            self.image_h5 = h5py.File(image_h5_path, 'r')
        
        vocab = kwargs.pop('vocab')
        mode = kwargs.pop('mode', 'prefix')
        
        max_samples = kwargs.pop('max_samples', None)
        question_h5_path = kwargs.pop('question_h5')
        image_idx_start_from = kwargs.pop('image_idx_start_from', None)
        print('Reading questions from ', question_h5_path)
        with h5py.File(question_h5_path, 'r') as question_h5:
            self.dataset = ClevrDataset(question_h5, self.feature_h5, vocab, mode,
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
        return [question_batch, image_batch, feat_batch, answer_batch, program_seq_batch, program_struct_batch]
        