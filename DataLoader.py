#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:52:45 2019

@author: jacob
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import Program_funcs as P
import random
import os

def dataset_to_tensor(dataset, mask=None):
    arr = np.asarray(dataset, dtype=np.int64)
    if mask is not None:
        arr = arr[mask]
    tensor = torch.LongTensor(arr)
    return tensor

class ClevrDataset(Dataset):
    def __init__(self, question_path, feature_path, vocab, mode='prefix', 
                 balanced_n=None, oversample=None, index_list_path=None,
                 image_h5=None, max_samples=None, hr_path=None,
                 image_idx_start_from=None):
        mode_choices = ['prefix', 'postfix']
        if mode not in mode_choices:
            raise ValueError('Invalid mode "%s"' % mode)
    
        self.mode = mode
        self.max_samples = max_samples
        self.balanced_n = balanced_n
        self.oversample = oversample
        
        self.index_list = None
        if index_list_path is not None:
            self.index_list = np.load(index_list_path, allow_pickle=True)
            self.index_list = self.index_list[()]
        if self.index_list is not None:
            self.categories = len(self.index_list.keys())
               
        self.vocab = vocab
        
        self.feature_path = feature_path
        self.hr_path = hr_path
        
        questions = np.load(question_path, allow_pickle=True)
        mask = None

        self.all_questions = dataset_to_tensor(questions[()]['questions'], mask)
        self.answers = dataset_to_tensor(questions[()]['answers'], mask)
        self.image_idxs = questions[()]['image_idxs'], mask
        self.programs = None
        if 'programs' in questions[()].keys():
            self.programs = dataset_to_tensor(questions[()]['programs'], mask)

        self.eval_index = 0
        self.eval = False

        if self.balanced_n is not None:
            if self.index_list is None:
                raise ValueError('Must provide index list')
            else:
                self.sample_list = []
                for index in self.index_list.keys():
                    indexs = random.sample(self.index_list[index], self.balanced_n)
                    self.sample_list.append(indexs)
                for _ in range(self.max_samples - len(self.sample_list)):
                    while True:
                        i = random.randint(0,len(self.all_questions)-1)
                        if i in self.sample_list:
                            continue
                        else:
                            self.sample_list.append(i)
                            break
        elif self.oversample:
            if self.index_list is None:
                raise ValueError('Must provide index list')
            else:
                max_n = 0
                for index in self.index_list.keys():
                    if len(self.index_list[index]) > max_n:
                        max_n = len(self.index_list[index])
                self.sample_list = list(range(max_n*self.categories))
                for i in range(self.categories):
                    current_list = self.index_list[index]
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
                
        if image_idx_start_from is not None:
            all_image_idxs = np.asarray(questions['image_idxs'])
            mask = all_image_idxs >= image_idx_start_from
          
    def __getitem__(self, i):
        if self.eval:
            index = self.eval_index
            self.eval_index += 1
            if self.eval_index == len(self.all_questions):
                self.done = True
        else:
            index = self.sample_list[i] 

        question = self.all_questions[index]
        answer = self.answers[index]
        print(self.image_idxs[()])
        print(index)
        image_idx = self.image_idxs[index]
        if self.programs is not None:
            program_seq = self.programs[index]
        
        feats = torch.load(self.feature_path + image_idx)
        feats = torch.FloatTensor(feats)
        
        image = None
        #Implement image loader here if needed

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
        program, I = self.get_program(question)
        return (question, image, feats, answer, program_seq, program_json, index, self.done, program, I)
    
    def __len__(self):
        return len(self.sample_list)
    
    def eval_mode(self):
        self.eval = True
        self.eval_index = 0
        self.done = False
        
    def get_program(self, question):
        q_name = '-'.join(str(int(e)) for e in question if e != 0)
        if os.path.exists(self.hr_path+q_name):
            programs_list = os.listdir(self.hr_path+q_name)
            if programs_list[0][-4:] == 'MAPO':
                length = len(programs_list)
                programs = torch.zeros(length, 30)
                for i in range(length):  
                    p_name = os.listdir(self.hr_path + q_name)[i]
                    program = torch.load(self.hr_path + q_name + '/' + p_name)
                    programs[i] = program
                return programs, False
            else:
                p_name = os.listdir(self.hr_path + q_name)[0]
                program = torch.load(self.hr_path + q_name + '/' + p_name)
                return program, True
        else:
            return None, False
    
class ClevrDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if 'question_path' not in kwargs:
            raise ValueError('Must give question path')
        if 'feature_path'  not in kwargs:
            raise ValueError('Must give feature path')
        if 'vocab' not in kwargs:
            raise ValueError('Must give vocab')
        index_list_path = kwargs.pop('path_to_index', None)
        oversample = kwargs.pop('oversample', None)
        balanced_n = kwargs.pop('balanced_n', None)
        feature_path = kwargs.pop('feature_path')
        hr_path = kwargs.pop('high_reward_path', None)
        
        self.image_h5 = None
        if 'image_h5' in kwargs:
            image_h5_path = kwargs.pop('image_h5')
            #Implement image loader here if needed
                   
        vocab = kwargs.pop('vocab')
        mode = kwargs.pop('mode', 'prefix')
        
        max_samples = kwargs.pop('max_samples', None)
        question_path = kwargs.pop('question_path')
        image_idx_start_from = kwargs.pop('image_idx_start_from', None)
        self.dataset = ClevrDataset(question_path, feature_path, vocab, mode,
                                    balanced_n=balanced_n, oversample=oversample,
                                    index_list_path=index_list_path, image_h5=self.image_h5,
                                    max_samples=max_samples, hr_path=hr_path,
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
        done = transposed[7]
        programs = transposed[8]
        I = transposed[9]
        return [question_batch, image_batch, feat_batch, answer_batch, program_seq_batch, program_struct_batch, i, done, programs, I]






















       
#class MyClevrDataLoader(DataLoader):
#    def __init__(self, **kwargs):
#        if 'question_h5' not in kwargs:
#            raise ValueError('Must give question_q5')
#        if 'feature_h5'  not in kwargs:
#            raise ValueError('Must give feature_h5')
#        if 'vocab' not in kwargs:
#            raise ValueError('Must give vocab')
#
#        self.oversample, self.balanced_n = False, None
#        self.shuffle = False
#        super(MyClevrDataLoader, self)
#        
#        if 'model' in kwargs:
#            model = kwargs.pop('model')
#            self.shuffle = kwargs.pop('shuffle')
#            self.balanced_n = kwargs.pop('balanced_n')
#            self.path_to_index = kwargs.pop('path_to_index')
#            if model == 'PG' and self.balanced_n is not None:
#                if self.balanced_n > 24:
#                    self.balanced_n = 24
#                    print('Number of samples for each answer key is > 24, setting it to 24')
#                with open(self.path_to_index) as f:
#                    self.index_list = json.load(f)
#            elif model == 'EE' and 'oversample' in kwargs:
#                self.oversample = kwargs.pop('oversample')
#                with open(self.path_to_index) as f:
#                    self.index_list = json.load(f)
#
#        feature_h5_path = kwargs.pop('feature_h5')
#        self.feature_h5 = h5py.File(feature_h5_path, 'r')
#        self.batch_size = kwargs.pop('batch_size')
#        self.image_h5 = None
#        if 'image_h5' in kwargs:
#            image_h5_path = kwargs.pop('image_h5')
#            self.image_h5 = h5py.File(image_h5_path, 'r')
#                    
#        self.vocab = kwargs.pop('vocab')
#        self.mode = kwargs.pop('mode', 'prefix')
#        
#        self.max_samples = kwargs.pop('max_samples', None)
#        self.question_h5_path = kwargs.pop('question_h5')
#        self.image_idx_start_from = kwargs.pop('image_idx_start_from', None)
#        
#        mode_choices = ['prefix', 'postfix']
#        if self.mode not in mode_choices:
#            raise ValueError('Invalid mode "%s"' % self.mode)
#        
#        mask = None
#        if self.image_idx_start_from is not None:
#            all_image_idxs = np.asarray(self.question_h5['image_idxs'])
#            mask = all_image_idxs >= self.image_idx_start_from
#
#        self.question_h5 = h5py.File(self.question_h5_path, 'r')
#        
#        self.all_questions = dataset_to_tensor(self.question_h5['questions'], mask)
#        self.all_image_idxs = dataset_to_tensor(self.question_h5['image_idxs'], mask)
#        self.all_programs = None
#        if 'programs' in self.question_h5:
#            self.all_programs = dataset_to_tensor(self.question_h5['programs'], mask)
#        self.all_answers = dataset_to_tensor(self.question_h5['answers'], mask)
#        
#        self.index = 0
#        self.mode = 'Train'
#        
#        if self.balanced_n is not None and model == 'PG':
#            self.sample_list = []
#            for i in range(len(self.index_list)-4):
#                indexs = random.sample(self.index_list[str(i+4)], self.balanced_n)
#                for index in indexs:
#                    self.sample_list.append(index)
#            for _ in range(self.max_samples - len(self.sample_list)):
#                while True:
#                    i = random.randint(0,len(self.all_questions)-1)
#                    if i in self.sample_list:
#                        continue
#                    else:
#                        self.sample_list.append(i)
#                        break
#        elif self.oversample and model == 'EE':
#            max_n = 0
#            for i in range(28):
#                if len(self.index_list[str(i+4)]) > max_n:
#                    max_n = len(self.index_list[str(i+4)])
#            self.sample_list = list(range(max_n*28))
#            for i in range(28):
#                current_list = self.index_list[str(i+4)]
#                max_len = len(current_list)
#                k = 0
#                for j in range(max_n):
#                    self.sample_list[(i*max_n)+j] = current_list[k]
#                    k += 1
#                    if k % max_len == 0:
#                        k = 0           
#                          
#        else:
#            if self.max_samples is not None:
#                self.sample_list = random.sample(range(len(self.all_questions)), self.max_samples)
#            else:
#                self.sample_list = list(range(len(self.all_questions)))
#        if self.shuffle:
#            self.shuffle_samples()
#            
#        self.max_index = len(self.sample_list)
#        if self.max_index > len(self.all_questions):
#            self.max_eval_index = len(self.all_questions)
#        else:
#            self.max_eval_index = self.max_index
#        
#        self.questions = torch.zeros(self.batch_size,46).long()
#        self.feats = torch.zeros(self.batch_size,1024,14,14) 
#        self.answers = torch.zeros(self.batch_size,1).long()
#        self.indexs = torch.zeros(self.batch_size,1).long()
#        self.program_seq = torch.zeros(self.batch_size,30).long()
#        self.images = torch.zeros(self.batch_size)
#        self.program_struct = []
#        
#    def batch(self):
#        if self.mode == 'Train':
#            for j in range(self.batch_size):
#                index = self.sample_list[self.index]
#                self.index += 1
#                
#                question = self.all_questions[index]
#                image_idx = self.all_image_idxs[index]
#                answer = self.all_answers[index]
#                program_seq = None
#                
#                if self.all_programs is not None:
#                    program_seq = self.all_programs[index]
#                    
#                image = None
#                if self.image_h5 is not None:
#                    image = self.image_h5['images'][image_idx]
#                    image = torch.FloatTensor(np.asarray(image, dtype=np.float32))
#                
#                feats = self.feature_h5['features'][image_idx]
#                feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))
#                
#                program_json = None
#                if program_seq is not None:
#                    program_json_seq = []
#                    for fn_idx in program_seq:
#                        fn_str = self.vocab['program_idx_to_token'][fn_idx.item()]
#                        if fn_str == '<START>' or fn_str == '<END>':
#                            continue
#                        fn = P.str_to_function(fn_str)
#                        program_json_seq.append(fn)
#                    if self.mode == 'prefix':
#                        program_json = P.prefix_to_list(program_json_seq)
#                    elif self.mode == 'postfix':
#                        program_json = P.postfix_to_list(program_json_seq)
#                
#                self.questions[j] = question
#                self.feats[j] = feats 
#                self.answers[j] = answer
#                self.indexs[j] = index
#                self.program_seq[j][0:len(program_seq)] = program_seq
#                self.program_struct.append(program_json)
#                
#                if self.index == self.max_index:
#                    self.index = 0
#                    self.shuffle_samples()
#                   
#            return [self.questions, self.images, self.feats, self.answers, self.program_seq, self.program_struct, self.indexs]
#        elif self.mode == 'eval':
#            for j in range(self.batch_size):
#                if len(self.sample_list) != len(self.all_questions):
#                    index = self.sample_list[self.eval_index]
#                else:
#                    index = self.eval_index
#                self.eval_index += 1
#                
#                question = self.all_questions[index]
#                image_idx = self.all_image_idxs[index]
#                answer = self.all_answers[index]
#                program_seq = None
#                
#                if self.all_programs is not None:
#                    program_seq = self.all_programs[index]
#                    
#                image = None
#                if self.image_h5 is not None:
#                    image = self.image_h5['images'][image_idx]
#                    image = torch.FloatTensor(np.asarray(image, dtype=np.float32))
#                
#                feats = self.feature_h5['features'][image_idx]
#                feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))
#                
#                program_json = None
#                if program_seq is not None:
#                    program_json_seq = []
#                    for fn_idx in program_seq:
#                        fn_str = self.vocab['program_idx_to_token'][fn_idx.item()]
#                        if fn_str == '<START>' or fn_str == '<END>':
#                            continue
#                        fn = P.str_to_function(fn_str)
#                        program_json_seq.append(fn)
#                    if self.mode == 'prefix':
#                        program_json = P.prefix_to_list(program_json_seq)
#                    elif self.mode == 'postfix':
#                        program_json = P.postfix_to_list(program_json_seq)
#                
#                self.questions[j] = question
#                self.feats[j] = feats 
#                self.answers[j] = answer
#                self.indexs[j] = index
#                self.program_seq[j][0:len(program_seq)] = program_seq
#                self.program_struct.append(program_json)
#            
#                if self.eval_index == self.max_eval_index:
#                    self.mode = 'Train'
#                    return [self.questions[:j+1], self.images[:j+1], self.feats[:j+1],
#                            self.answers[:j+1], self.program_seq[:j+1], 
#                            self.program_struct[:j+1], self.indexs[:j+1], True]
#                
#            return [self.questions, self.images, self.feats, self.answers, 
#                    self.program_seq, self.program_struct, self.indexs, False]
#        
#    def eval_mode(self):
#        self.mode = 'eval'
#        self.eval_index = 0
#    
#    def shuffle_samples(self):
#        np.random.shuffle(self.sample_list)
#        
#class ClevrDataset(Dataset):
#    def __init__(self, question_h5_path, feature_h5_path, vocab, mode='prefix', 
#                 balanced_n=None, oversample=None, index_list=None,
#                 image_h5=None, max_samples=None, hr_path=None,
#                 image_idx_start_from=None, question_categories=28):
#        mode_choices = ['prefix', 'postfix']
#        if mode not in mode_choices:
#            raise ValueError('Invalid mode "%s"' % mode)
#        self.image_h5 = image_h5
#        self.vocab = vocab
#        self.feature_h5_path = feature_h5_path
#        self.mode = mode
#        self.max_samples = max_samples
#        self.balanced_n = balanced_n
#        self.oversample = oversample
#        self.index_list = index_list
#        qc = question_categories
#        self.eval_index = 0
#        self.eval = False
#        self.hr_path = hr_path
#        self.question_h5_path = question_h5_path
#        self.question_h5 = h5py.File(question_h5_path, 'r')
#
#        mask = None
#        self.all_questions = dataset_to_tensor(self.question_h5['questions'], mask)
#
#       
#        if self.balanced_n is not None:
#            if self.index_list is None:
#                raise ValueError('Must provide index list')
#            else:
#                self.sample_list = []
#                for i in range(len(self.index_list)-4):
#                    indexs = random.sample(self.index_list[str(i+4)], self.balanced_n)
#                    self.sample_list.append(indexs)
#                for _ in range(self.max_samples - len(self.sample_list)):
#                    while True:
#                        i = random.randint(0,len(self.all_questions)-1)
#                        if i in self.sample_list:
#                            continue
#                        else:
#                            self.sample_list.append(i)
#                            break
#        elif self.oversample:
#            if self.index_list is None:
#                raise ValueError('Must provide index list')
#            else:
#                max_n = 0
#                for i in range(qc):
#                    if len(self.index_list[str(i+4)]) > max_n:
#                        max_n = len(self.index_list[str(i+4)])
#                self.sample_list = list(range(max_n*qc))
#                for i in range(qc):
#                    current_list = self.index_list(str(i+4))
#                    max_len = len(current_list)
#                    k = 0
#                    for j in range(max_n):
#                        self.sample_list[(i*max_n)+j] = current_list[k]
#                        k += 1
#                        if k % max_len == 0:
#                            k = 0                             
#        else:
#            if self.max_samples is not None:
#                self.sample_list = random.sample(range(len(self.all_questions)), self.max_samples)
#            else:
#                self.sample_list = list(range(len(self.all_questions)))
#                
##        if image_idx_start_from is not None:
##            all_image_idxs = np.asarray(question_h5['image_idxs'])
##            mask = all_image_idxs >= image_idx_start_from
#          
#    def __getitem__(self, i):
#        mask = None
#        with h5py.File(self.feature_h5_path, 'r') as feature_h5:
#            with h5py.File(self.question_h5_path, 'r') as all_qs:
#                all_questions = dataset_to_tensor(all_qs['questions'], mask)
#                all_image_idxs = dataset_to_tensor(all_qs['image_idxs'], mask)
#                self.all_programs = None
#                if 'programs' in self.question_h5:
#                    all_programs = dataset_to_tensor(all_qs['programs'], mask)
#                all_answers = dataset_to_tensor(all_qs['answers'], mask)        
#                
#                if self.eval:
#                    index = self.eval_index
#                    self.eval_index += 1
#                    if self.eval_index == len(all_questions):
#                        self.done = True
#                else:
#                    index = self.sample_list[i]        
#                question = all_questions[index]
#                image_idx = all_image_idxs[index]
#                answer = all_answers[index]
#                program_seq = None
#                if all_programs is not None:
#                    program_seq = self.all_programs[index]
#                    
#                image = None
#                if self.image_h5 is not None:
#                    image = self.image_h5['images'][image_idx]
#                    image = torch.FloatTensor(np.asarray(image, dtype=np.float32))
#                
#                feats = feature_h5['features'][image_idx]
#                feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))
#                
#                program_json = None
#                if program_seq is not None:
#                    program_json_seq = []
#                    for fn_idx in program_seq:
#                        fn_str = self.vocab['program_idx_to_token'][fn_idx.item()]
#                        if fn_str == '<START>' or fn_str == '<END>':
#                            continue
#                        fn = P.str_to_function(fn_str)
#                        program_json_seq.append(fn)
#                    if self.mode == 'prefix':
#                        program_json = P.prefix_to_list(program_json_seq)
#                    elif self.mode == 'postfix':
#                        program_json = P.postfix_to_list(program_json_seq)
#                program, I = self.get_program(question)
#                return (question, image, feats, answer, program_seq, program_json, index, self.done, program, I)
#    
#    def __len__(self):
#        return len(self.sample_list)
#    
#    def eval_mode(self):
#        self.eval = True
#        self.eval_index = 0
#        self.done = False
#        
#    def get_program(self, question):
#        q_name = '-'.join(str(int(e)) for e in question if e != 0)
#        if os.path.exists(self.hr_path+q_name):
#            programs_list = os.listdir(self.hr_path+q_name)
#            if programs_list[0][-4:] == 'MAPO':
#                length = len(programs_list)
#                programs = torch.zeros(length, 30)
#                for i in range(length):  
#                    p_name = os.listdir(self.hr_path + q_name)[i]
#                    program = torch.load(self.hr_path + q_name + '/' + p_name)
#                    programs[i] = program
#                return programs, False
#            else:
#                p_name = os.listdir(self.hr_path + q_name)[0]
#                program = torch.load(self.hr_path + q_name + '/' + p_name)
#                return program, True
#        else:
#            return None, False
#    
#class ClevrDataLoader(DataLoader):
#    def __init__(self, **kwargs):
#        if 'question_h5' not in kwargs:
#            raise ValueError('Must give question_q5')
#        if 'feature_h5'  not in kwargs:
#            raise ValueError('Must give feature_h5')
#        if 'vocab' not in kwargs:
#            raise ValueError('Must give vocab')
#        path_to_index = kwargs.pop('path_to_index', None)
#        oversample = kwargs.pop('oversample', None)
#        balanced_n = kwargs.pop('balanced_n', None)
#        feature_h5_path = kwargs.pop('feature_h5')
#        hr_path = kwargs.pop('high_reward_path', None)
#        
#        self.image_h5 = None
#        if 'image_h5' in kwargs:
#            image_h5_path = kwargs.pop('image_h5')
#            self.image_h5 = h5py.File(image_h5_path, 'r')
#                   
#        vocab = kwargs.pop('vocab')
#        mode = kwargs.pop('mode', 'prefix')
#        
#        if path_to_index is not None:
#            index_list = json.load(open(path_to_index))
#        else:
#            index_list = None
#        
#        max_samples = kwargs.pop('max_samples', None)
#        question_h5_path = kwargs.pop('question_h5')
#        image_idx_start_from = kwargs.pop('image_idx_start_from', None)
#        self.dataset = ClevrDataset(question_h5_path, feature_h5_path, vocab, mode,
#                                    balanced_n=balanced_n, oversample=oversample,
#                                    index_list=index_list, image_h5=self.image_h5,
#                                    max_samples=max_samples, hr_path=hr_path,
#                                    image_idx_start_from=image_idx_start_from)
#        kwargs['collate_fn'] = self.clevr_collate
#        super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)
#    
#    def close(self):
#        if self.image_h5 is not None:
#            self.image_h5.close()
#        if self.feature_h5 is not None:
#            self.feature_h5.close()
#    
#    def __enter__(self):
#        return self
#            
#    def __exit__(self, exc_type, exc_value, traceback):
#        self.close()
#    
#    def clevr_collate(self, batch):
#        transposed = list(zip(*batch))
#        question_batch = default_collate(transposed[0])
#        image_batch = transposed[1]
#        if any(img is not None for img in image_batch):
#            image_batch = default_collate(image_batch)
#        feat_batch = transposed[2]
#        if any(f is not None for f in feat_batch):
#            feat_batch = default_collate(feat_batch)
#        answer_batch = default_collate(transposed[3])
#        program_seq_batch = transposed[4]
#        if transposed[4][0] is not None:
#            program_seq_batch = default_collate(transposed[4])
#        program_struct_batch = transposed[5]
#        i = transposed[6]
#        done = transposed[7]
#        programs = transposed[8]
#        I = transposed[9]
#        return [question_batch, image_batch, feat_batch, answer_batch, program_seq_batch, program_struct_batch, i, done, programs, I]