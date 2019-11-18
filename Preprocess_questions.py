#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:44:06 2019

@author: jacob
"""

import json
import h5py
import numpy as np
from Preprocess_funcs import tokenize, build_vocab, encode, program_to_str
##Args

path = "../Data/questions/"
questions = "val" #train / val / test / all

input_vocab = '../Data/vocab/vocab.json' #Path to json vocab we want to expand (empty '' = create new)
output_vocab = '../Data/vocab/vocab.json' #Dumb path for new expanded vocab
expand_vocab = 'y' #Are we expanding already existing vocab? y/n

h5_output = '../Data/h5py/questions_h5py_'+questions

unk_threshold = 1 #Word must occur this many times to not be <UNK>
punct_to_remove = [';',',']
punct_to_keep = ['?','.']
keep_program = 'y' #Keep supervision for programs? y/n

allow_unk = False #Allow <UNK> entries in vocab? True/False
mode = 'prefix'

##Preprocessing
if questions == 'all':
    questions_list = ['train','val','test']
else:
    questions_list = [questions]
for questions in questions_list:    
    print("Loading data")
    path_to_q = path + 'CLEVR_' + questions + '_questions.json' 
    with open(path_to_q, 'r') as f:
        questions = json.load(f)['questions']
    if input_vocab == '' or expand_vocab == 'y':
        print('Building new vocab')
      
        if 'answer' in questions[0]: #Is test set or not
            answer_token_to_idx = build_vocab(
                    (q['answer'] for q in questions))
        
        question_token_to_idx = build_vocab(
                (q['question'] for q in questions),
                min_token_count=unk_threshold, punct_to_keep=punct_to_keep,
                punct_to_remove=punct_to_remove)
        
        if keep_program == 'y':
            all_program_strs = []
            for q in questions:
                if 'program' not in q:
                    continue
                program_str = program_to_str(q['program'], mode)
                if program_str is not None:
                    all_program_strs.append(program_str)
            program_token_to_idx = build_vocab(all_program_strs)
        
        if 'answer' not in questions[0]:
            print('Dataset is test, not encoding programs')
            keep_program = 'n'
            
        if keep_program == 'y':
            vocab = {'question_token_to_idx':question_token_to_idx,
                     'program_token_to_idx': program_token_to_idx,
                     'answer_token_to_idx': answer_token_to_idx}
        elif 'answer' in questions[0]:
            vocab = {'question_token_to_idx': question_token_to_idx,
                     'answer_token_to_idx': answer_token_to_idx}
        else:
            vocab = {'question_token_to_idx': question_token_to_idx}
        
    if input_vocab != '':
        print('Loading vocab')
        if expand_vocab == 'y':
            new_vocab = vocab
        with open(input_vocab, 'r') as f:
            vocab = json.load(f)
        if expand_vocab == 'y':
            num_new_words = 0
            for word in new_vocab['question_token_to_idx']:
                if word not in vocab['question_token_to_idx']:
                    print('Found new word %s' % word)
                    idx = len(vocab['question_token_to_idx'])
                    vocab['question_token_to_idx'][word] = idx
                    num_new_words += 1
            print('Expanded vocab with %d new words' % num_new_words)
    
    if output_vocab != '':
        with open(output_vocab, 'w') as f:
            json.dump(vocab, f)
        
        
    
    print('Encoding the data')
    questions_encoded = []
    programs_encoded = []
    orig_idxs = []
    image_idxs = []
    answers = []
    
    for orig_idx, q in enumerate(questions):
        question = q['question']
    
        orig_idxs.append(orig_idx)
        image_idxs.append(q['image_index'])
        question_tokens = tokenize(question, punct_to_keep=punct_to_keep,
                                   punct_to_remove=punct_to_remove)
        question_encoded = encode(question_tokens, vocab['question_token_to_idx'],
                                   allow_unk=allow_unk)
        questions_encoded.append(question_encoded)
        
        if keep_program == 'y':
            if 'program' in q:
                program = q['program']
                program_str = program_to_str(program, mode)
                program_tokens = tokenize(program_str)
                program_encoded = encode(program_tokens, vocab['program_token_to_idx'])
                programs_encoded.append(program_encoded)
                
        if 'answer' in q:
            answers.append(vocab['answer_token_to_idx'][q['answer']])
            
    #Pad the encoded Q's to equal length
    max_q_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_q_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])
            
    if len(programs_encoded) > 0:
        max_p_length = max(len(x) for x in programs_encoded)
        for pe in programs_encoded:
            while len(pe) < max_p_length:
                pe.append(vocab['program_token_to_idx']['<NULL>'])
        
    #Dump file as h5
    print('Writing output')
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    programs_encoded = np.asarray(programs_encoded, dtype=np.int32)
    
    print('Encoded questions shape')
    print(questions_encoded.shape)
    
    if keep_program == 'y':
        print('Encoded programs shape')
        print(programs_encoded.shape)
    
    
    with h5py.File(h5_output, 'w') as f:
        f.create_dataset('questions', data=questions_encoded)
        f.create_dataset('image_idxs', data=np.asarray(image_idxs))
        f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))
        
        if len(programs_encoded) > 0:
            f.create_dataset('programs', data=programs_encoded)
        
        if len(answers) > 0:
            f.create_dataset('answers', data=np.asarray(answers))
                
                
            
            
            
            
            
            