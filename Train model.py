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
import torch.multiprocessing as mp
from MAPO_workers import MAPO

import re
import datetime

import time




"""
Some network params are set in LSTM_Model.py and Module_Net.py
"""

#%% Setup Params
if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    
    # Start from an existing checkpoint
    parser.add_argument('--pg_start_from', default="../Data/models/PG_15k.pt")
    parser.add_argument('--ee_start_from', default="../Data/models/EE_15k.pt")
    parser.add_argument('--mapo', default=True)
    
    # What type of model to use and which parts to train
    parser.add_argument('--model_type', default='PG+EE',
            choices=['PG', 'EE', 'PG+EE'])
    parser.add_argument('--train_pg', default=1, type=int)
    parser.add_argument('--train_ee', default=1, type=int)
    parser.add_argument('--balanced_loss', default=True)
    
    #Training length
    parser.add_argument('--num_iterations', default=200000, type=int)
    parser.add_argument('--epochs', default=0, type=int) 
    #If 0 epochs we use num_iterations to determine training length
    parser.add_argument('--break_after', default=10, type=int)
    #If val has not improved after break_after checks, we early stop
    
    parser.add_argument('--info', default=False)
    #Do you want all info or minimal?
    
    #Samples and shuffeling
    parser.add_argument('--num_train_samples', default=None, type=int) #None = All
    parser.add_argument('--num_val_samples', default=15000, type=int)
    parser.add_argument('--shuffle_train_data', default=False, type=int)
    
    #Bloom Filter
    parser.add_argument('--bf_est_ele', default=10**6, type=int)
    parser.add_argument('--bf_false_pos_rate', default=0.01, type=float)
    parser.add_argument('--bf_load_path', default='../Data/bloom_filters')
    
    parser.add_argument('--bloom_percentage', default=0.05, type=float)
    
    #MAPO
    parser.add_argument('--MAPO_use_GPU', default=0, type=int) 
    parser.add_argument('--MAPO_sample_argmax', default=0, type=int)
    parser.add_argument('--MAPO_score_cutoff', default=0, type=float)
    #parser.add_argument('--MAPO_alpha', default=0.1, type=float) #TODO Not in use?
    parser.add_argument('--MAPO_check_bf', default=0.01, type=float)
    parser.add_argument('--MAPO_check_bf_argmax', default=False)
    parser.add_argument('--MAPO_split', default=2/3, type=float)
    #If --MAPO_split is None we do not assume program generator is correct
    
    #Datapaths
    parser.add_argument('--train_questions_h5', default='../Data/h5py/questions_h5py_train')
    parser.add_argument('--train_features_h5', default='../Data/h5py/img_features_h5py_train')
    parser.add_argument('--val_questions_h5', default='../Data/h5py/questions_h5py_val')
    parser.add_argument('--val_features_h5', default='../Data/h5py/img_features_h5py_val')
    parser.add_argument('--vocab_json', default='../Data/vocab/vocab.json') 
    parser.add_argument('--high_reward_path', default='../Data/high_reward_paths/')
    
    parser.add_argument('--checkpoint_path', default='../Data/models/')
    
    #Dataloader params
    parser.add_argument('--feature_dim', default='1024,14,14')
    parser.add_argument('--loader_num_workers', type=int, default=0)
    
    # LSTM options
    parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
    parser.add_argument('--rnn_hidden_dim', default=256, type=int)
    parser.add_argument('--rnn_num_layers', default=2, type=int)
    parser.add_argument('--rnn_dropout', default=0, type=float)
    parser.add_argument('--length_output', default=30, type=int)
    
    # Module net options
    parser.add_argument('--module_stem_num_layers', default=2, type=int)
    parser.add_argument('--module_stem_batchnorm', default=0, type=int)
    parser.add_argument('--module_dim', default=128, type=int)
    parser.add_argument('--module_residual', default=1, type=int)
    parser.add_argument('--module_batchnorm', default=0, type=int)
    
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
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--reward_decay', default=0.99, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    
    # Output options
    parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
    parser.add_argument('--record_loss_every', type=int, default=1)
    parser.add_argument('--checkpoint_every', default=1000, type=int)
    
    #%%Train loop
    args = parser.parse_args()
    vocab = func.load_vocab(args.vocab_json)
    if args.mapo:
        cpu_count = mp.cpu_count()
        if args.model_type != 'PG+EE':
            raise KeyError("MAPO can only train with both PG and EE. Change model_type")
    
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
    
    if args.model_type == "PG":
        if args.num_train_samples is not None:
            model_name = args.model_type+'_'+str(int(args.num_train_samples)//1000)+'k'
        else:
            model_name = args.model_type+'_'+'700k'
    elif args.mapo == True:
        model_name = 'MAPO'+'_'+re.findall(r'[0-9]+',args.pg_start_from)[0]+'k'
    else:
        model_name = args.model_type+'_'+re.findall(r'[0-9]+',args.pg_start_from)[0]+'k'
        
    
    args.checkpoint_path = args.checkpoint_path + model_name
    
    if args.num_train_samples == None:
        args.num_train_samples = 10**9
        
    if not args.mapo:
        train_loader = ClevrDataLoader(**train_loader_kwargs)    
    val_loader = ClevrDataLoader(**val_loader_kwargs)
        
    
    #Indent here
    program_generator, pg_kwargs, pg_optimizer = None, None, None
    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    
    pg_best_state, ee_best_state = None, None
    
    #Set up model
    program_generator, pg_kwargs = func.get_program_generator(vocab, args)
    pg_optimizer = torch.optim.Adam(program_generator.parameters(),
                                        lr=args.learning_rate)
    if args.info:
        print('Here is the program generator:')
        print(program_generator)
    if args.model_type == 'EE' or args.model_type == 'PG+EE':
        execution_engine, ee_kwargs = func.get_execution_engine(vocab, args)
        ee_optimizer = torch.optim.Adam(execution_engine.parameters(),
                                        lr=args.learning_rate)
        if args.info:
            print('Here is the execution engine:')
            print(execution_engine)
        
    if args.balanced_loss:
        class_counter = torch.Tensor([0,0,0,0,48658,57907,24,29855,15256,7418,
                                      3615,1662,647,259,105,7873,7926,21014,7731,
                                      21051,7786,7778,31342,31353,143713,7973,
                                      7797,31477,31587,21087,7974,139121])
        #TODO: Automate counter process in preprocessing
        I = class_counter != 0
        min_class = min(class_counter[I])
        class_weights = min_class / class_counter
        class_weights = class_weights.cuda()

    if args.balanced_loss:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss().cuda()
        

    stats = {'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
             'train_accs':[], 'val_accs': [], 'val_accs_ts': [],
             'best_val_acc': -1, 'best_model_t': 0, 'epoch': []}
    
    t, epoch, reward_moving_avg = 0,0,0
    
    if args.mapo:    
        if args.MAPO_use_GPU == 0:
            if args.info:
                print('MAPO will use %d CPUs' % cpu_count-2)
            func.set_mode('eval', [program_generator, execution_engine])
            execution_engine.share_memory()
            program_generator.share_memory()
            processes = []
            que = mp.Queue()
            loader_que = mp.Queue()
            
            for cpu in range(cpu_count-3):
                p = mp.Process(target=MAPO, args=(args, program_generator.cpu(),
                                                  execution_engine.cpu(), 
                                                  loader_que, vocab, que, cpu))
         
                p.start() 
                if args.info:
                    print('MAPO worker %s spawned' % str(cpu))
                processes.append(p)
            
            p = mp.Process(target=func.h5py_loader, args=(train_loader_kwargs, 
                                                          loader_que, 2*cpu_count))
            p.start()
            if args.info:
                print('Clevr dataloader spawned')
            processes.append(p)
                                
        else:
            raise KeyError('MAPO does not support actors on GPUs')
            #TODO: Implement GPU MAPO?
    _loss = []
    break_counter = 0
    func.set_mode('train', [program_generator, execution_engine])
    while True:
        if epoch == args.epochs and args.epochs != 0:
            if args.mapo:
                for p in processes:
                    p.terminate()
            break
        elif t == args.num_iterations and args.epochs == 0:
            if args.mapo:
                for p in processes:
                    p.terminate()
            break
        elif (args.break_after is not None) and (break_counter >= args.break_after):
            if args.mapo:
                for p in processes:
                    p.terminate()
            break
        epoch += 1
        if args.info:
            print('Starting epoch %d' % epoch)
        if not args.mapo:
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
                    programs_pred = program_generator.reinforce_sample(questions_var)
                    scores = execution_engine(feats_var, programs_pred)
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
                    
                    if args.train_ee == 1:
                        ee_optimizer.zero_grad()
                        loss.backward()
                        ee_optimizer.step()
                        
                    if args.train_pg == 1:
                        pg_optimizer.zero_grad()
                        program_generator.reinforce_backward(centered_reward.cuda())
                        pg_optimizer.step()
                        
                _loss.append(loss.item())
    
                if t % args.record_loss_every == 0:
                    stats['train_losses'].append(sum(_loss)/len(_loss))
                    stats['train_losses_ts'].append(t)
                    if reward is not None:
                        stats['train_rewards'].append(reward)
            
                if t % args.checkpoint_every == 0:
                    if args.info:
                        print('Calculating accuracy')
                    train_acc = func.check_accuracy(args, program_generator,
                                                execution_engine, train_loader)
                    val_acc = func.check_accuracy(args, program_generator,
                                              execution_engine, val_loader)
                    stats['train_accs'].append(train_acc)
                    stats['val_accs'].append(val_acc)
                    stats['val_accs_ts'].append(t)
                    stats['epoch'].append(epoch)
                    if val_acc > stats['best_val_acc']:
                        stats['best_val_acc'] = val_acc
                        stats['best_model_t'] = t
                        best_pg_state = func.get_state(program_generator)
                        best_ee_state = func.get_state(execution_engine)
                        break_counter = 0
                        improved_val = "+"
                    else: 
                        break_counter += 1
                        improved_val = "-"
                    print('%s - %d - %f \t Train acc: %.4f \t Val acc: %.4f (%s)'  \
                          % (model_name, t, sum(_loss)/len(_loss), train_acc, val_acc, improved_val))
                    _loss = []
                        
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
                    torch.save(checkpoint, args.checkpoint_path+'.pt')
                    del checkpoint['program_generator_state']
                    del checkpoint['execution_engine_state']
                    with open(args.checkpoint_path + '.json', 'w') as f:
                        json.dump(checkpoint, f)
                    
                if args.break_after is not None:
                    if break_counter >= args.break_after:
                        break
                if args.num_iterations is not None:    
                    if t == args.num_iterations and args.epochs == 0:
                        break
    
        #MAPO
        else:
            idle_t = []
            epoch = True #TODO Make this check something
            while epoch:
                t += 1
                i = 0
                if args.MAPO_split is None:
                    MAPO_amount = 1
                else:
                    MAPO_amount = int(1/args.MAPO_split)
                for j in range(MAPO_amount):
                    MAPO_feats_var = torch.zeros(args.batch_size, 1024, 14, 14) #Todo Automate 1024x14x14
                    MAPO_programs_pred = torch.zeros(args.batch_size, args.length_output)
                    MAPO_ans = torch.zeros(args.batch_size)
                    MAPO_q = torch.zeros(args.batch_size, 46) #Automate q length
                    MAPO_m_out = []
                    MAPO_m_probs = []
                    MAPO_m_m = []
                    t = time.time()
                    while i < args.batch_size:
                        q_tmp, feat_tmp, program_tmp, ans_tmp = que.get()
                        MAPO_feats_var[i], MAPO_programs_pred[i] = feat_tmp.clone(), program_tmp.clone()
                        MAPO_q[i], MAPO_ans[i] = ans_tmp.clone(), q_tmp.clone()
                        del feat_tmp, program_tmp, q_tmp, ans_tmp
                        #We need to release this memory back to the MAPO workers
                        m_out, m_probs, m_m = program_generator.program_to_probs(MAPO_q[i], MAPO_programs_pred[i],
                                                                                 args.temperature)
                        MAPO_m_out.append(m_out)
                        MAPO_m_probs.append(m_probs)
                        MAPO_m_m.append(m_m)
                        i += 1
                        print(i)
                    idle_t.append(time.time()-t)
                    
                    scores = execution_engine(MAPO_feats_var, MAPO_programs_pred)
    
                    _, preds = scores.data.cpu().max(1)
                    I = (preds == MAPO_ans)
                    raw_reward = I.float()
                    reward_moving_avg *= args.reward_decay
                    reward_moving_avg+= (1.0 - args.reward_decay) * raw_reward.mean()
                    centered_reward = raw_reward - reward_moving_avg
                    
                    _loss.append(loss_fn(scores, MAPO_ans).item())
                    loss = loss_fn(scores[I,:], MAPO_ans[I])
                                        
                    if args.train_ee:
                        ee_optimizer.zero_grad()
                        loss.backwards()
                        ee_optimizer.step()
                    pg_optimizer.zero_grad()
                    program_generator.reinforce_backward_MAPO(MAPO_m_out, MAPO_m_probs, MAPO_m_m, centered_reward)
                    pg_optimizer.step()
                
                if args.MAPO_split is not None:
                    batch = loader_que.get()
                    questions, _, feats, answers, programs, _ = batch
                    questions_var = Variable(questions.cuda())
                    feats_var = Variable(feats.cuda())
                    answers_var = Variable(answers.cuda())
                    programs_pred = program_generator.reinforce_sample(questions_var)
                    scores = execution_engine(feats_var, programs_pred)
                    loss = loss_fn(scores, answers_var)
                    ee_optimizer.zero_grad()  
                    loss.backward()
                    ee_optimizer.step()
                 
                _loss.append(loss.item())
                
                if t % args.record_loss_every == 0:
                    stats['train_losses'].append(sum(_loss)/len(_loss))
                    stats['train_losses_ts'].append(t)
                    if reward is not None:
                        stats['train_rewards'].append(reward)
        
                
                if t % args.checkpoint_every == 0:
                    if args.info:
                        print('Calculating accuracy')
                    train_acc = func.check_accuracy(args, program_generator,
                                                execution_engine, train_loader)
                    val_acc = func.check_accuracy(args, program_generator,
                                              execution_engine, val_loader)
                    stats['train_accs'].append(train_acc)
                    stats['val_accs'].append(val_acc)
                    stats['val_accs_ts'].append(t)
                    stats['epoch'].append(epoch)
    
                    if val_acc > stats['best_val_acc']:
                        stats['best_val_acc'] = val_acc
                        stats['best_model_t'] = t
                        best_pg_state = func.get_state(program_generator)
                        best_ee_state = func.get_state(execution_engine)
                        break_counter = 0
                        improved_val = "+"
                    else:
                        break_counter += 1
                        improved_val = "-"
                    idle = sum(idle_t)/len(idle_t)
                    print('%s - %d - %f \t Train acc: %.4f \t Val acc: %.4f (%s) \t Idle: %s'  \
                          % (model_name, t, sum(_loss)/len(_loss), train_acc, 
                             val_acc, improved_val, str(datetime.timedelta(seconds=idle))[:-7]))
                    _loss = []
                    
                    checkpoint = {'args': args.__dict__,
                                  'program_generator_kwargs': pg_kwargs,
                                  'program_generator_state': best_pg_state,
                                  'execution_engine_kwargs': ee_kwargs,
                                  'execution_engine_state': best_ee_state,
                                  'vocab': vocab}
                    for k, v in stats.items():
                        checkpoint[k] = v
                    if args.info:
                        print('Saving checkpoint to %s' % args.checkpoint_path)
                    torch.save(checkpoint, args.checkpoint_path)
                    del checkpoint['program_generator_state']
                    del checkpoint['execution_engine_state']
                    with open(args.checkpoint_path + '.json', 'w') as f:
                        json.dump(checkpoint, f)
                        
                if break_counter >= args.break_after:
                    break
                if args.num_iterations is not None:        
                    if t == args.num_iterations and args.epochs == 0:
                        break 
    
    print('Model %s is done, performing last accuracy check and saving model' % model_name)
    print('Model %s trained for %d epochs' % (model_name, epoch))
    if args.info:
        print('Calculating accuracy')
    train_acc = func.check_accuracy(args, program_generator,
                                    execution_engine, train_loader)
    print('Train accuracy for %s is: %.4f' % (model_name, train_acc))
    val_acc = func.check_accuracy(args, program_generator,
                                  execution_engine, val_loader)
    print('Val accuracy for %s is: %.4f' % (model_name, val_acc))
    stats['train_accs'].append(train_acc)
    stats['val_accs'].append(val_acc)
    stats['val_accs_ts'].append(t)
    
    if val_acc > stats['best_val_acc']:
        stats['best_val_acc'] = val_acc
        stats['best_model_t'] = t
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
    if args.info:
        print('Saving checkpoint to %s' % args.checkpoint_path+'.pt')
    torch.save(checkpoint, args.checkpoint_path+'.pt')
    del checkpoint['program_generator_state']
    del checkpoint['execution_engine_state']
    with open(args.checkpoint_path + '.json', 'w') as f:
        json.dump(checkpoint, f)
        

            
        
        



    