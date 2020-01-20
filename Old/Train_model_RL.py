#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:27:24 2019

@author: jacob
"""
import Functions as func
import torch
from torch.autograd import Variable
import argparse
from DataLoader import ClevrDataLoader
import torch.multiprocessing as mp
from MAPO_workers import MAPO_CPU, MAPO_GPU, MAPO_GPU_EE

import time

#class_counter = torch.Tensor([0,0,0,0,48658,57907,24,29855,15256,7418,
#                              3615,1662,647,259,105,7873,7926,21014,7731,
#                              21051,7786,7778,31342,31353,143713,7973,
#                              7797,31477,31587,21087,7974,139121])


#TODO: program_generator.sample_non_high_reward samples random sample instead of non reward one
#TODO: Add oversampling
#TODO: Add random load order in batch loader
#TODO: Add MAPO GPU EE
#TODO: Add trainloader capability for check_func
#TODO: Add smarter novel path method since program is predicted in reverse
#TODO: Sample initial strongly labelled dataset 'smarter'
train_loader = 1 
#%% Setup Params
if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    
    # Start from an existing checkpoint
    parser.add_argument('--pg_start_from', default="../Data/models/PG_10k.pt")
    parser.add_argument('--ee_start_from', default="../Data/models/EE_10k.pt")
    parser.add_argument('--mapo', default=True)
    
    # What type of model to use and which parts to train
    parser.add_argument('--model_type', default='PG+EE',
            choices=['PG', 'EE', 'PG+EE'])
    parser.add_argument('--train_pg', default=1, type=int)
    parser.add_argument('--train_ee', default=1, type=int)
    parser.add_argument('--oversample', default=False)
    
    #Training length
    parser.add_argument('--num_iterations', default=200000, type=int)
    parser.add_argument('--epochs', default=0, type=int) 
    #If 0 epochs we use num_iterations to determine training length
    parser.add_argument('--break_after', default=10, type=int)
    #If val has not improved after break_after checks, we early stop
    
    parser.add_argument('--info', default=False)
    #Do you want all info or minimal?
    
    #Samples and shuffeling
    parser.add_argument('--num_train_samples', default=None, type=int) 
    #If None we load all examples
    parser.add_argument('--num_val_samples', default=15000, type=int)
    parser.add_argument('--shuffle_train_data', default=False, type=int)
    
    #Bloom Filter
    parser.add_argument('--bf_est_ele', default=10**3, type=int)
    parser.add_argument('--bf_false_pos_rate', default=0.01, type=float)
    parser.add_argument('--bf_load_path', default='../Data/bloom_filters')
    
    #MAPO
    parser.add_argument('--MAPO_auto_cleanup', default=1, type=int)
    parser.add_argument('--MAPO_use_GPU', default=0, type=int) 
    parser.add_argument('--MAPO_max_cpus', default=20, type=int)
    parser.add_argument('--MAPO_qsize', default=320, type=int)
    parser.add_argument('--MAPO_sample_argmax', default=True)
    parser.add_argument('--MAPO_check_bf', default=0.01, type=float)
    parser.add_argument('--MAPO_check_bf_argmax', default=False)
    parser.add_argument('--MAPO_rate', default=2, type=float)
    #The rate of times we train the PG vs the EE
    
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
    parser.add_argument('--checkpoint_every', default=50, type=int)
    
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
    
    model_name = func.auto_namer(args)
    args.checkpoint_path = args.checkpoint_path + model_name
             
    val_loader = ClevrDataLoader(**val_loader_kwargs)
        
    program_generator, pg_kwargs, pg_optimizer = None, None, None
    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    
    best_pg_state, best_ee_state = None, None
    
    #Set up model
    if args.MAPO_use_GPU:
        if torch.cuda.device_count() < 2:
            raise Exception('MAPO_use_GPU requires 2 GPUs')
    program_generator, pg_kwargs = func.get_program_generator(vocab, args)
    pg_optimizer = torch.optim.Adam(program_generator.parameters(),
                                        lr=args.learning_rate)

    if args.model_type != 'PG' or args.mapo:
        execution_engine, ee_kwargs = func.get_execution_engine(vocab, args)
        ee_optimizer = torch.optim.Adam(execution_engine.parameters(),
                                        lr=args.learning_rate)
        
    loss_fn = torch.nn.CrossEntropyLoss().cuda()      
   
    stats = {'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
             'train_accs':[], 'val_accs': [], 'val_accs_ts': [],
             'best_val_acc': -1, 'best_model_t': 0, 'epoch': []}
    
    t, epoch, reward_moving_avg = 0,0,0
    
    _loss = []
    break_counter = 0
    
    #%% Non MAPO    
    if not args.mapo:
        cont = True
        train_loader = ClevrDataLoader(**train_loader_kwargs)    
        func.set_mode('train', [program_generator, execution_engine])
        while cont:
            epoch += 1
            if args.info:
                print('Starting epoch %d' % epoch)
            for batch in train_loader:
                t += 1
                questions, _, feats, answers, programs, _ = batch
                questions_var = Variable(questions.cuda())
                feats_var = Variable(feats.cuda())
                answers_var = Variable(answers.cuda())
                #Train PG
                if args.model_type == 'PG':
                    programs_var = Variable(programs.cuda())
                    pg_optimizer.zero_grad()
                    loss = program_generator(questions_var, programs_var)
                    loss.backward()
                    pg_optimizer.step()
                #Train EE
                elif args.model_type == 'EE':
                    programs_pred = program_generator.reinforce_sample(questions_var)
                    scores = execution_engine(feats_var, programs_pred)
                    ee_optimizer.zero_grad()  
                    loss = loss_fn(scores, answers_var)
                    loss.backward()
                    ee_optimizer.step()
                #Joint training
                elif args.model_type == 'PG+EE':
                    reward = None
                    programs_pred = program_generator.reinforce_sample(questions_var)   
                    scores = execution_engine(feats_var, programs_pred)
                    loss = loss_fn(scores, answers_var)
                    _, preds = scores.data.cpu().max(1)
                    raw_reward = (preds == answers).float()
                    reward_moving_avg *= args.reward_decay
                    reward_moving_avg += (1.0 - args.reward_decay) * raw_reward.mean()
                    centered_reward = raw_reward - reward_moving_avg
                    reward = centered_reward.mean().item()
                    
                    if args.train_ee == 1:
                        ee_optimizer.zero_grad()
                        loss.backward()
                        ee_optimizer.step()
                        
                    if args.train_pg == 1:
                        pg_optimizer.zero_grad()
                        program_generator.reinforce_backward(centered_reward.cuda())
                        pg_optimizer.step()
                        
                _loss.append(loss.item())
    
                stats['train_losses'].append(sum(_loss)/len(_loss))
                stats['train_losses_ts'].append(t)
                if reward is not None:
                    stats['train_rewards'].append(reward)
            
                if t % args.checkpoint_every == 0:
                    stats, break_counter, best_pg_state, best_ee_state =\
                    func.checkpoint_func(args, program_generator, execution_engine, 
                                         train_loader, val_loader, t, epoch, stats,
                                         model_name, _loss, pg_kwargs, ee_kwargs, 
                                         vocab, break_counter, best_pg_state, best_ee_state)
                if args.break_after is not None:
                    if break_counter >= args.break_after:
                        print('Model %s is done training' % model_name)
                        cont = False
                        break
                if args.num_iterations is not None:    
                    if t == args.num_iterations:
                        cont = False
                        print('Model %s is done training - performing last accuracy check' % model_name)
                        stats, break_counter, best_pg_state, best_ee_state =\
                        func.checkpoint_func(args, program_generator, execution_engine, 
                                             train_loader, val_loader, t, epoch, stats,
                                             model_name, _loss, pg_kwargs, ee_kwargs, 
                                             vocab, break_counter, best_pg_state, best_ee_state)
                        break
    
    #%%MAPO
    else: 
        if args.MAPO_auto_cleanup:
            func.clean_up(args)                        
        cpu_count = mp.cpu_count()
        if args.MAPO_max_cpus is not None:
            cpu_count = min(cpu_count, args.MAPO_max_cpus)
        if args.info:
            print('MAPO will use %d CPUs' % cpu_count)
        func.set_mode('eval', [program_generator, execution_engine])
        execution_engine.share_memory()
        program_generator.share_memory()
        processes = []
        pg_que = mp.Queue()
        loader_que = mp.Queue()
        ee_que = mp.Queue()
        skip_que = mp.Queue()
        p = mp.Process(target=func.MAPO_loader, args=(train_loader_kwargs, 
                                                      loader_que, ee_que, 
                                                      skip_que, args.MAPO_qsize))
        p.start()
        processes.append(p)
        if args.info:
            print('Clevr dataloader spawned')
            
        if args.MAPO_use_GPU == 0:
            for cpu in range(cpu_count-2):
                p = mp.Process(target=MAPO_CPU, args=(args, program_generator.cpu(),
                                                  execution_engine.cpu(), 
                                                  loader_que, vocab, pg_que,
                                                  skip_que, cpu))
         
                p.start() 
                processes.append(p)
                if args.info:
                    print('MAPO worker %s spawned' % str(cpu))
            
        else:
            MAPO_que = mp.Queue()
            for cpu in range(cpu_count-3):
                p = mp.Process(target=MAPO_GPU, args=(args, program_generator.cpu(),
                                                     execution_engine.cpu(), 
                                                     loader_que, vocab, MAPO_que,
                                                     skip_que, cpu))
                p.start()
                processes.append(p)
                if args.info:
                    print('MAPO worker %s spawned' % str(cpu))
            p = mp.Process(target=MAPO_GPU_EE, args=(args, program_generator.cuda(1),
                                                     execution_engine.cuda(1),
                                                     MAPO_que, pg_que, ee_que,
                                                     skip_que))
            p.start()
            processes.append(p)
            if args.info:
                print('Evaluator for MAPO spawned')
        
        func.set_mode('train', [program_generator, execution_engine])
        program_generator.cuda()
        execution_engine.cuda()
        idle_t = []
        cont = True
        while cont:
            t += 1
            for _ in range(args.MAPO_rate):
                i = 0
                j = 0
                feats_var = torch.zeros(args.batch_size, 1024, 14, 14).cuda() #TODO: Automate 1024x14x14
                programs_pred = torch.zeros(args.batch_size, args.length_output).long().cuda()
                ans = torch.zeros(args.batch_size).long().cuda()
                q = torch.zeros(args.batch_size, 46).long().cuda() #TODO: Automate q length
                time_ = time.time()
                while i < args.batch_size:
                    q_tmp, feat_tmp, program_tmp, ans_tmp = pg_que.get()
                    feats_var[i,:], programs_pred[i,:] = feat_tmp.cuda().clone(), program_tmp.cuda().clone()
                    q[i,:], ans[i] = q_tmp.cuda().clone(), ans_tmp.cuda().clone()
                    del feat_tmp, program_tmp, q_tmp, ans_tmp
                    #We need to release this memory back to the MAPO workers
                    i += 1
                idle_t.append(time.time()-time_)        
                m_out, m_probs, m_m = program_generator.program_to_probs(q, programs_pred,
                                                                         args.temperature)
                scores = execution_engine(feats_var, programs_pred)
                _, preds = scores.data.cpu().max(1)
                I = (preds == ans.cpu()).long()
                raw_reward = I.float()
                reward_moving_avg *= args.reward_decay
                reward_moving_avg+= (1.0 - args.reward_decay) * raw_reward.mean()
                centered_reward = raw_reward - reward_moving_avg
                reward = centered_reward.mean().item()
                _loss.append(loss_fn(scores, ans).item())
                I = I.cuda()
                loss = loss_fn(scores[I,:], ans[I])
                if args.train_ee:
                    ee_optimizer.zero_grad()
                    loss.backward()
                    ee_optimizer.step()
                pg_optimizer.zero_grad()
                program_generator.reinforce_backward_MAPO(m_out, m_probs, m_m, centered_reward.cuda())
                pg_optimizer.step()                
            
            while j < args.batch_size:
                q_tmp, feat_tmp, ans_tmp = ee_que.get()
                feats_var[j,:] = feat_tmp.cuda().clone()
                q[j,:], ans[j] = q_tmp.cuda().clone(), ans_tmp.cuda().clone()
                del feat_tmp, q_tmp, ans_tmp
                #We need to release this memory back to the MAPO workers
                j += 1
                
            ee_optimizer.zero_grad()  
            programs_pred = program_generator.reinforce_sample(q)
            scores = execution_engine(feats_var, programs_pred)
            loss = loss_fn(scores, ans)
            loss.backward()
            ee_optimizer.step()
            _loss.append(loss.item())
            
            stats['train_losses'].append(sum(_loss)/len(_loss))
            stats['train_losses_ts'].append(t)
            stats['train_rewards'].append(reward)
            
            if t % args.checkpoint_every == 0:
                stats, break_counter, best_pg_state, best_ee_state =\
                func.checkpoint_func(args, program_generator, execution_engine, 
                                     train_loader, val_loader, t, epoch, stats,
                                     model_name, _loss, pg_kwargs, ee_kwargs, 
                                     vocab, break_counter, best_pg_state, best_ee_state)
            if args.break_after is not None:
                if break_counter >= args.break_after:
                    print('Model %s is done training' % model_name)
                    cont = False
                    break
            if args.num_iterations is not None:    
                if t == args.num_iterations:
                    cont = False
                    print('Model %s is done training - performing last accuracy check' % model_name)
                    stats, break_counter, best_pg_state, best_ee_state =\
                    func.checkpoint_func(args, program_generator, execution_engine, 
                                         train_loader, val_loader, t, epoch, stats,
                                         model_name, _loss, pg_kwargs, ee_kwargs, 
                                         vocab, break_counter, best_pg_state, best_ee_state)
                    break            
                
                
