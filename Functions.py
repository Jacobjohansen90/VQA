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
from DataLoader import ClevrDataset
import h5py
import os
import shutil

#Auto model namer
def auto_namer(model, args):
    if args.PG_num_of_each is not None:
        model_name = model+'_'+str(int(args.num_PG_samples)//1000)+'k_'+str(args.PG_num_of_each)+'_each'
    elif args.num_PG_samples is not None:
        model_name = model+'_'+str(int(args.num_PG_samples)//1000)+'k'
    else:
        model_name = model+'_'+'700k'
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
    if not path.endswith('.pt'):
        path += '.pt'    
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    model = Seq2Seq(**kwargs)
    model.load_state_dict(state)
    
    return model, kwargs
          
def get_program_generator(vocab, args, multi_gpu=True):
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
    if multi_gpu:
        if args.multi_GPU and torch.cuda.device_count() > 1:
            pg = torch.nn.DataParallel(pg)
            if args.info:
                print('Program Generator will use ', torch.cuda.device_count(), 'GPUs')
    if args.info:
        print('Here is the program generator:')
        print(pg)
    return pg, kwargs

#Execution Engine
def load_execution_engine(path, info):
    if not path.endswith('.pt'):
        path += '.pt'
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    kwargs = checkpoint['execution_engine_kwargs']
    state = checkpoint['execution_engine_state']
    model = ModuleNet(info, **kwargs)
    model.load_state_dict(state)
    return model, kwargs

def get_execution_engine(vocab, args, multi_gpu=True):
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
    if multi_gpu:
        if args.multi_GPU and torch.cuda.device_count() > 1:
            ee = torch.nn.DataParallel(ee)
            if args.info:
                print('Execution Engine will use ', torch.cuda.device_count(), 'GPUs')    
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
            
def check_accuracy(args, model, program_generator, execution_engine, loader, mode):
    set_mode('eval', [program_generator, execution_engine])
    if model != 'MAPO':
        loader.eval_mode()
    elif model == 'MAPO' and mode != 'train':
        loader.eval_mode()
    num_correct, num_samples = 0,0
    done = False
    while not done:
        if model == 'MAPO' and mode == 'train':
            questions, feats, answers, done = loader.get()
        else:
            questions, _, feats, answers, programs, _, _, done = loader.batch()
        scores = None
        programs_pred = program_generator.module.reinforce_sample(questions.cuda())
        if model == 'PG':
            I1 = (programs_pred != 0)
            I2 = (programs != 0)
            for i in range(programs_pred.shape[0]):
                num_samples += 1
                if len(programs_pred[i][I1[i]].cpu()) == len(programs[i][I2[i]][1:]):
                    if all(programs_pred[i][I1[i]].cpu() == programs[i][I2[i]][1:]):
                        num_correct += 1

        else: 
            scores = execution_engine(feats.cuda(), programs_pred)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == answers.squeeze(1)).sum()
            num_samples += preds.size(0)
    set_mode('train', [program_generator, execution_engine])
    acc = float(num_correct) / num_samples
    acc = round(acc, 4)
    return acc
    
def checkpoint_func(args, model, program_generator, execution_engine,
                    train_loader, val_loader, t, epoch, stats,
                    model_name, pg_loss, ee_loss, pg_kwargs, ee_kwargs, 
                    vocab, break_counter, best_pg_state, best_ee_state,
                    checkpoint_path):
    if args.info:
        print('Calculating accuracy')
    if model == 'PG':
        _loss = pg_loss
    elif model == 'EE':
        _loss = ee_loss
    elif model == 'MAPO':
        _loss = pg_loss + ee_loss
    train_acc = check_accuracy(args, model, program_generator,
                                execution_engine, train_loader, 'train')
    val_acc = check_accuracy(args, model, program_generator,
                              execution_engine, val_loader, 'val')
    stats['train_accs'].append(train_acc)
    stats['val_accs'].append(val_acc)
    stats['val_accs_ts'].append(t)
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
    if round(train_acc,4) == 1:
        break_counter = 10**6
    print('%s - %d - %f \t Train acc: %.4f \t Val acc: %.4f (%s)'  \
          % (model_name, t, sum(_loss)/len(_loss), train_acc, val_acc, improved_val))
        
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
    torch.save(checkpoint, checkpoint_path+'.pt')
    del checkpoint['program_generator_state']
    del checkpoint['execution_engine_state']
    with open(checkpoint_path + '.json', 'w') as f:
        json.dump(checkpoint, f)
    return stats, break_counter, best_pg_state, best_ee_state

#MAPO Functions  
def MAPO_loader(args, hr_list, MAPO_que, pg_que, ee_que, skip_que, eval_que, vocab):
    loader_kwargs = {
    'question_h5': args.train_questions_h5,
    'feature_h5': args.train_features_h5,
    'vocab':vocab,
    'batch_size':args.batch_size,
    'shuffle': args.shuffle_train_data,
    'max_samples': args.num_train_samples,
    'num_workers': args.loader_num_workers,
    'balanced_n':args.PG_num_of_each,
    'oversample':args.oversample,
    'path_to_index': args.path_to_index_file,
    'model': 'MAPO'}
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
    
    question_h5 = h5py.File(question_h5_path, 'r')
    dataset = ClevrDataset(False, False, question_h5, feature_h5, vocab, mode,
                           image_h5=image_h5, max_samples=max_samples,
                           image_idx_start_from=image_idx_start_from)

    non_hr_list = list(range(len(dataset)))
        
    for i in non_hr_list:
        if i in hr_list:
            non_hr_list.remove(i)

    i = 0; j = 0; k = 0; l = 0
    done = False
    q = torch.zeros(args.batch_size, 46).long()
    f = torch.zeros(args.batch_size, 1024, 14, 14)
    a = torch.zeros(args.batch_size, 1).long()
    while True:
        if eval_que.qsize() < 30:
            for i in range(args.batch_size):
                sample = dataset[l]
                q[i] = sample[0]
                f[i] = sample[2]
                a[i] = sample[3]
                l += 1
                if l == len(dataset):
                    done = True
                    l = 0
                    counter = i+1
                    break
            if not done:
                eval_que.put((q,f,a,done))
            elif done:
                eval_que.put((q[:counter], f[:counter], a[:counter], done))
                done = False

        if MAPO_que.qsize() < args.MAPO_qsize:
            index = non_hr_list[i]
            i+= 1
            MAPO_que.put(dataset[index])
            if i % len(non_hr_list)== 0:
                i = 0
        if ee_que.qsize() < args.MAPO_qsize:
            index = non_hr_list[j]
            j += 1
            ee_que.put(dataset[index])
            if j % len(non_hr_list) == 0:
                j = 0
        if pg_que.qsize() < args.MAPO_qsize:
            index = hr_list[k]
            k += 1
            pg_que.put(dataset[index])
            if k % len(hr_list) == 0:
                k = 0

        for _ in range(skip_que.qsize()):
            index = skip_que.get()
            #We need these additional checks if either list is small, as a 
            #sample might be in the que more than once, and thus returned
            #here in the skip que more than once. 
            if index < 0:
                if abs(index) in hr_list:
                    hr_list.remove(abs(index))
                if abs(index) not in non_hr_list:
                    non_hr_list.append(abs(index))
            else:
                if index not in hr_list:
                    hr_list.append(index)
                if index in non_hr_list:
                    non_hr_list.remove(index)
                
            
def make_HR_paths(args, pg, ee, loader):
    hr_list = []
    loader.eval_mode()
    done = False
    while not done:
        q, _, feat, ans, _, _, j, done = loader.batch()
        program_pred = pg.reinforce_sample(q.cuda())
        scores = ee(feat.cuda(), program_pred)
        _, preds = scores.data.cpu().max(1)
        for i in range(preds.size(0)):
            if preds[i] == ans[i]:
                hr_list.append(int(j[i]))
                q_name = '-'.join(str(int(e)) for e in q[i] if e != 0)
                q_name = q_name + '/'
                p_name = '-'.join(str(int(e)) for e in program_pred[i] if e != 0)
                path = args.high_reward_path + q_name
                if not os.path.exists(path):
                    os.mkdir(path)
                torch.save(program_pred[i], path+p_name)
    return hr_list

def update_hr_paths(args, program_pred, questions, index, skip_que, remove):
    hr_folder = args.high_reward_path
    bf_folder = args.bf_load_path
    for i in range(questions.size(0)):
        q_name = '-'.join(str(int(e)) for e in questions[i] if e != 0)
        p_name = '-'.join(str(int(e)) for e in program_pred[i] if e != 0)
        if remove:
            os.remove(hr_folder+q_name+'/'+p_name+'.pt')
            skip_que.put(-index[i])

        else:
            skip_que.put(index[i])
            if not os.path.exists(hr_folder+q_name):
                os.makedirs(hr_folder+q_name)
                torch.save(program_pred[i], hr_folder+q_name+'/'+p_name+'.pt')
                os.remove(bf_folder+q_name)

def load_hr_program(args, question):
    q_name = '-'.join(str(int(e)) for e in question if e != 0)
    p_name = os.listdir(args.high_reward_path + q_name)[0]
    return torch.load(args.high_reward_path + q_name + '/' + p_name)

def clean_up(args):
    bf_path = args.bf_load_path + '/'
    hr_path = args.high_reward_path 
    shutil.rmtree(bf_path)
    shutil.rmtree(hr_path)
    os.mkdir(hr_path)
    os.mkdir(bf_path)
           

    
    

        