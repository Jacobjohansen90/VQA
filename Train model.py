#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:27:24 2019

@author: jacob
"""
import json


import Functions as func
import torch
from torch.autograd import Variable
import argparse
from DataLoader import ClevrDataLoader


"""
All network params are set in Models.py
"""

#%% Setup Params
#TODO Clean up

parser = argparse.ArgumentParser()

#Datapaths
parser.add_argument('--train_questions_h5', default='../Data/h5py/questions_h5py_train')
parser.add_argument('--train_features_h5', default='../Data/h5py/img_features_h5py_train')
parser.add_argument('--val_questions_h5', default='../Data/h5py/questions_h5py_val')
parser.add_argument('--val_features_h5', default='../Data/h5py/img_features_h5py_val')
parser.add_argument('--vocab_json', default='../Data/vocab/vocab.json') 
parser.add_argument('--checkpoint_path', default='../Data/checkpoint.pt')
#
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--loader_num_workers', type=int, default=1)

#Samples and shuffeling
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=10000, type=int)
parser.add_argument('--shuffle_train_data', default=True, type=int)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='EE',
        choices=['PG', 'EE', 'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--pg_start_from', default="../Data/checkpoint_PG.pt")
parser.add_argument('--exec_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# LSTM options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument('--cnn_pooling', default='maxpool2',
        choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
        choices=['maxpool2', 'maxpool4', 'none'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0, type=float)

# Optimization options
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=20000, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)

# Output options
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--print_loss_every', type=int, default=50)
parser.add_argument('--record_loss_every', type=int, default=1)
parser.add_argument('--checkpoint_every', default=2500, type=int)

#%%Train loop

args = parser.parse_args()
vocab = func.load_vocab(args.vocab_json)

train_loader_kwargs = {
        'question_h5': args.train_questions_h5,
        'feature_h5': args.train_features_h5,
        'vocab':vocab,
        'batch_size':args.batch_size,
        'shuffle': args.shuffle_train_data,
        'max_samples': args.num_train_samples,
        'num_workers': args.loader_num_workers}

val_loader_kwargs = {
        'question_h5': args.val_questions_h5,
        'feature_h5': args.val_features_h5,
        'vocab': vocab,
        'batch_size': args.batch_size,
        'max_samples': args.num_val_samples,
        'num_workers': args.loader_num_workers}

with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
     ClevrDataLoader(**val_loader_kwargs) as val_loader:
    program_generator, pg_kwargs, pg_optimizer = None, None, None
    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    
    pg_best_state, ee_best_state = None, None
    
    #Set up model
    if args.model_type == 'PG' or args.model_type == 'PG+EE':
        program_generator, pg_kwargs = func.get_program_generator(vocab, args)
        pg_optimizer = torch.optim.Adam(program_generator.parameters(),
                                        lr=args.learning_rate)
        print('Here is the program generator:')
        print(program_generator)
    if args.model_type == 'EE' or args.model_type == 'PG+EE':
        execution_engine, ee_kwargs = func.get_execution_engine(vocab, args)
        ee_optimizer = torch.optim.Adam(execution_engine.parameters(),
                                        lr=args.learning_rate)
        print('Here is the execution engine:')
        print(execution_engine)
        
    
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    
    stats = {'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
             'train_accs':[], 'val_accs': [], 'val_accs_ts': [],
             'best_val_acc': -1, 'model_t': 0}
    
    t, epoch, reward_moving_avg = 0,0,0
    
    func.set_mode('train', [program_generator, execution_engine])
    
    print('Train loader has %d samples' % len(train_loader.dataset))
    print('Validation loader has %d samples' % len(val_loader.dataset))
    _loss = []
    while t < args.num_iterations:
        epoch += 1
        print('Starting epoch %d' % epoch)
        for batch in train_loader:
            t += 1
            questions, _, feats, answers, programs, _ = batch
            questions_var = Variable(questions.cuda())
            feats_var = Variable(feats.cuda())
            answers_var = Variable(answers.cuda())
            if programs[0] is not None:
                programs_var = Variable(programs.cuda())
            
            reward = None
            if args.model_type == 'PG':
                pg_optimizer.zero_grad()
                loss = program_generator(questions_var, programs_var)
                loss.backward()
                pg_optimizer.step()
            elif args.model_type == 'EE':
                ee_optimizer.zero_grad()
                scores = execution_engine(feats_var, programs_var)
                loss = loss_fn(scores, answers_var)
                loss.backward()
                ee_optimizer.step()
            elif args.model_type == 'PG+EE':
                programs_pred = program_generator.reinforce_sample(questions_var)
                scores = execution_engine(feats_var, programs_pred)
                
                loss = loss_fn(scores, answers_var)
                _, preds = scores.data.cpu().max(1)
                raw_reward = (preds == answers).float()
                reward_moving_avg *= args.reward_decay
                reward_moving_avg += (1.0 - args.reward_decay) * raw_reward.mean()
                centered_reward = raw_reward - reward_moving_avg
                
                if args.train_execution_engine == 1:
                    ee_optimizer.zero_grad()
                    loss.backward()
                    ee_optimizer.step()
                    
                if args.train_program_generator == 1:
                    pg_optimizer.zero_grad()
                    program_generator.reinforce_backward(centered_reward.cuda())
                    pg_optimizer.step()
            if t % args.record_loss_every == 0:
                stats['train_losses'].append(loss.data.item())
                stats['train_losses_ts'].append(t)
                if reward is not None:
                    stats['train_rewards'].append(reward)

            _loss.append(loss.item())

            if t % args.print_loss_every == 0:
                print(t, sum(_loss)/len(_loss))
                _loss = []
                
            if t % args.checkpoint_every == 0:
                print('Calculating accuracy')
                train_acc = func.check_accuracy(args, program_generator,
                                                execution_engine, train_loader)
                print('Train accuracy is: ', train_acc)
                val_acc = func.check_accuracy(args, program_generator,
                                              execution_engine, val_loader)
                print('Val accuracy is: ', val_acc)
                stats['train_accs'].append(train_acc)
                stats['val_accs'].append(val_acc)
                stats['val_accs_ts'].append(t)
                
                if val_acc > stats['best_val_acc']:
                    stats['best_val_acc'] = val_acc
                    stats['model_t'] = t
                    best_pg_state = func.get_state(program_generator)
                    best_ee_state = func.get_state(execution_engine)
                    
                checkpoint = {'args': args.__dict__,
                              'program_generator_kwargs': pg_kwargs,
                              'program_generator_state': best_pg_state,
                              'execution_engine_kwargs': ee_kwargs,
                              'execution_engine_state': best_ee_state,
                              'vocab': vocab}
                
                for k, v in stats.items():
                    checkpoint[k] = v
                print('Saving checkpoint to %s' % args.checkpoint_path)
                torch.save(checkpoint, args.checkpoint_path)
                del checkpoint['program_generator_state']
                del checkpoint['execution_engine_state']
                with open(args.checkpoint_path + '.json', 'w') as f:
                    json.dump(checkpoint, f)
                    
            if t == args.num_iterations:
                break
                
                            
                
                
                
    
        
        
    