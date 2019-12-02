#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:27:24 2019

@author: jacob
"""
import Functions as func
import torch
import argparse
from DataLoader import MyClevrDataLoader
import torch.multiprocessing as mp

#TODO: Add smarter strongly supervised sampling
#TODO: Add smarter novel path method since program is predicted in reverse
#TODO: Load stats properbly
#TODO: Create oversample.json during preprocessing
#%% Setup Params
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    parser.add_argument('--info', default=False)
    #Do you want all info or minimal?

    #Amount of times we train with postives and negatives examples in one pass
    parser.add_argument('--ee_train_count', default=1, type=int)
    parser.add_argument('--pg_train_count', default=1, type=int)    

    #Training length / early stopping
    parser.add_argument('--num_iterations', default=200000, type=int)
    parser.add_argument('--break_after', default=10, type=int)
    #If val has not improved after break_after checks, we early stop

    #Samples and shuffeling
    parser.add_argument('--shuffle_train_data', default=True, type=int)
    parser.add_argument('--num_PG_samples', default=5000, type=int)
    parser.add_argument('--PG_balanced_sampling', default=True)
    parser.add_argument('--PG_num_of_each', default=10, type=int) #No larger than 24
    parser.add_argument('--oversample', default=True)
    parser.add_argument('--num_train_samples', default=5000, type=int) 
    parser.add_argument('--num_val_samples', default=None, type=int)
    #If None we load all examples

    # Optimization options
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--reward_decay', default=0.99, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    
    # Output options
    parser.add_argument('--checkpoint_every', default=1000, type=int)

    # What type of model to use and which parts to train
    parser.add_argument('--model_type', default='MAPO',
            choices=['PG', 'EE', 'MAPO', 'all'])

    # Start from an existing checkpoint
    parser.add_argument('--pg_start_from', default='../Data/models/PG_5k.pt')
    parser.add_argument('--ee_start_from', default='../Data/models/EE_5k.pt')

    #Bloom Filter options
    parser.add_argument('--bf_est_ele', default=10**3, type=int)
    parser.add_argument('--bf_false_pos_rate', default=0.01, type=float)
    
    #MAPO
    parser.add_argument('--MAPO_clean_up', default=1, type=int)
    parser.add_argument('--MAPO_max_cpus', default=20, type=int)
    parser.add_argument('--MAPO_qsize', default=320, type=int)

    parser.add_argument('--MAPO_sample_argmax', default=False)
    parser.add_argument('--MAPO_check_bf_argmax', default=False)
    
    #Datapaths
    parser.add_argument('--train_questions_h5', default='../Data/h5py/questions_h5py_train')
    parser.add_argument('--train_features_h5', default='../Data/h5py/img_features_h5py_train')
    parser.add_argument('--val_questions_h5', default='../Data/h5py/questions_h5py_val')
    parser.add_argument('--val_features_h5', default='../Data/h5py/img_features_h5py_val')
    parser.add_argument('--vocab_json', default='../Data/vocab/vocab.json') 
    parser.add_argument('--high_reward_path', default='../Data/high_reward_paths/')    
    parser.add_argument('--checkpoint_path', default='../Data/models/')
    parser.add_argument('--bf_load_path', default='../Data/bloom_filters/')
    parser.add_argument('--oversampling_list', default='../Data/oversample.json')
    parser.add_argument('--path_to_index_file', default='../Data/ans_to_index.json')
    
    #Dataloader params
    parser.add_argument('--feature_dim', default='1024,14,14')
    parser.add_argument('--loader_num_workers', type=int, default=0)
    
    # Program generator (LSTM Model) options
    parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
    parser.add_argument('--rnn_hidden_dim', default=256, type=int)
    parser.add_argument('--rnn_num_layers', default=2, type=int)
    parser.add_argument('--rnn_dropout', default=0, type=float)
    parser.add_argument('--length_output', default=30, type=int)
    
    # Execution engine (Module net) options
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
    
    #%%Training loop
    args = parser.parse_args()
    vocab = func.load_vocab(args.vocab_json)
         
    val_loader_kwargs = {
            'question_h5': args.val_questions_h5,
            'feature_h5': args.val_features_h5,
            'vocab': vocab,
            'batch_size': args.batch_size,
            'max_samples': args.num_val_samples,
            'num_workers': args.loader_num_workers}

    val_loader = MyClevrDataLoader(**val_loader_kwargs)
    
    if args.model_type == 'all':
        model = ['PG', 'EE', 'MAPO']
    else:
        model = [args.model_type]

    
    for model_ in model:
        print('Training %s' % model_)
        #Setup current model        
        program_generator, pg_kwargs, pg_optimizer = None, None, None
        execution_engine, ee_kwargs, ee_optimizer = None, None, None
        
        best_pg_state, best_ee_state = None, None
        
        #Set amount of training samples
        if model_ == 'PG':
            num_train_samples = args.num_PG_samples
        else:
            num_train_samples = args.num_train_samples
        
        #Load previous trained models if needed
        if args.model_type == 'all':        
            if model_ == 'EE' and args.pg_start_from is None:
                args.pg_start_from = args.checkpoint_path
            if model_ == 'MAPO' and args.ee_start_from is None:
                args.ee_start_from = args.checkpoint_path
                
        program_generator, pg_kwargs = func.get_program_generator(vocab, args)
        pg_optimizer = torch.optim.Adam(program_generator.parameters(),
                                        lr=args.learning_rate)
        if model_ != 'PG':
            execution_engine, ee_kwargs = func.get_execution_engine(vocab, args)
            ee_optimizer = torch.optim.Adam(execution_engine.parameters(),
                                            lr=args.learning_rate)
            
        #Auto checkpointing        
        model_name = func.auto_namer(model_, args)
        args.checkpoint_path = args.checkpoint_path + model_name
        
        #Setup loss and train loader
        loss_fn = torch.nn.CrossEntropyLoss().cuda()      
        

                
        train_loader_kwargs = {
            'question_h5': args.train_questions_h5,
            'feature_h5': args.train_features_h5,
            'vocab':vocab,
            'batch_size':args.batch_size,
            'shuffle': args.shuffle_train_data,
            'max_samples': num_train_samples,
            'num_workers': args.loader_num_workers,
            'balanced': args.PG_balanced_sampling,
            'balanced_n':args.PG_num_of_each,
            'oversample':args.oversample,
            'path_to_index': args.path_to_index_file,
            'model':model_}
        
        train_loader = MyClevrDataLoader(**train_loader_kwargs)    
        
        
        if model_ == 'MAPO':
                     
            #Fill high reward buffer
            if args.MAPO_clean_up:
                func.clean_up(args)
            func.set_mode('eval', [program_generator, execution_engine])      
            hr_list = func.make_HR_paths(args, program_generator, execution_engine, train_loader)            
            
            #Spawn MAPO workers, dataloaders and bloom filter checkers          
            cpu_count = mp.cpu_count()
            execution_engine.share_memory()
            program_generator.share_memory()
            
            pg_que, ee_que, wait_que, skip_que, processes = func.spawn_MAPO(args, program_generator, 
                                                                            execution_engine, cpu_count,
                                                                            train_loader_kwargs, 
                                                                            vocab, hr_list)
            #Set model to GPU            
            program_generator.cuda()
            execution_engine.cuda()    

            #Placeholders for loading
            questions = torch.zeros(args.batch_size, 46).long()
            feats = torch.zeros(args.batch_size, 1024, 24, 24)
            answers = torch.zeros(args.batch_size).long()
            index = torch.zeros(args.batch_size).long()
            programs = torch.zeros(args.batch_size, 30).long()
        
        stats = {'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
                 'train_accs':[], 'val_accs': [], 'val_accs_ts': [],
                 'best_val_acc': -1, 'best_model_t': 0, 'epoch': []}
        
        t, epoch = 0,0
        pg_loss = []
        ee_loss = []
        break_counter = 0
        cont = True 
        
        func.set_mode('train', [program_generator, execution_engine])
        
        while cont:
            if model_ == 'PG':
                while True:
                    t += 1
                    questions, _, feats, answers, programs, _, _ = train_loader.batch()
                    pg_optimizer.zero_grad()
                    loss = program_generator(questions.cuda(), programs.cuda())
                    loss.backward()
                    pg_loss.append(loss.item())
                    pg_optimizer.step()
                   
                    if t % args.checkpoint_every == 0:
                        stats, break_counter, best_pg_state, best_ee_state =\
                        func.checkpoint_func(args, model_, program_generator, execution_engine, 
                                             train_loader, val_loader, t, epoch, stats,
                                             model_name, pg_loss, ee_loss, pg_kwargs, ee_kwargs, 
                                             vocab, break_counter, best_pg_state, best_ee_state)
                        pg_loss = []
                        
                    if break_counter >= args.break_after:
                        cont = False
                        break
                    
            elif model_ == 'EE':
                while True:
                    t += 1
                    questions, _, feats, answers, _, _, _ = train_loader.batch()
                    programs_pred = program_generator.reinforce_sample(questions.cuda())
                    scores = execution_engine(feats.cuda(), programs_pred.cuda())
                    ee_optimizer.zero_grad()
                    loss = loss_fn(scores, answers)
                    loss.backward()
                    ee_loss.append(loss.item())
                    ee_optimizer.step()

                    if t % args.checkpoint_every == 0:
                        stats, break_counter, best_pg_state, best_ee_state =\
                        func.checkpoint_func(args, program_generator, execution_engine, 
                                             train_loader, val_loader, t, epoch, stats,
                                             model_name, pg_loss, ee_loss, pg_kwargs, ee_kwargs, 
                                             vocab, break_counter, best_pg_state, best_ee_state)
                        ee_loss = []
                        
                    if break_counter >= args.break_counter:
                        cont = False
                        break

            elif model_ == 'MAPO':
                t += 1
                for _ in range(args.pg_train_count):
                    #Positive examples
                    i = 0
                    while i < args.batch_size:
                        questions[i], _, feats[i], answers[i], _, _, index[i] = pg_que.get()
                        programs[i] = func.load_hr_program(questions[i])
                        i += 1
                    #Train the PG
                    pg_optimizer.zero_grad()
                    loss = program_generator(questions.cuda(), programs.cuda())
                    loss.backward()
                    pg_loss.append(loss.item())
                    pg_optimizer.step()

                    #Check that all samples are positive else update positive list
                    scores = execution_engine(feats.cuda(), programs.cuda())
                    _, preds = scores.data.cpu().max(1)
                    I = (preds == answers)
                    if I.sum() != args.batch_size:
                        func.update_hr_paths(args, programs[~I], questions[~I], skip_que, index[~I], True)

                    #Train the EE
                    ee_optimizer.zero_grad()
                    loss = loss_fn(scores, answers)
                    loss.backward()
                    ee_loss.append(loss.item())
                    ee_optimizer.step()

                for _ in range(args.ee_train_count):
                    #Negative examples
                    j = 0
                    while j < args.batch_size:
                        questions[i], _, feats[i], answers[i], _, _, index[i] = ee_que.get()
                        j += 1
                    programs_pred = program_generator.reinforce_sample(questions.cuda())
                    scores = execution_engine(feats.cuda(), programs_pred.cuda())
                    _, preds = scores.data.cpu().max(1)
                    I = (preds == answers)
                    if I.sum() != 0:
                        func.update_hr_paths(args, programs_pred[I], questions[I], skip_que, index[I], False)
                    #Train the EE
                    ee_optimizer.zero_grad()
                    loss = loss_fn(scores, answers)
                    loss.backward()
                    ee_loss.append(loss.item())
                    ee_optimizer.step()
                        
                if t % args.checkpoint_every == 0:
                    #We need to make sure only one dataloader is laoding
                    #from the h5py files. Else we will get an error
                    wait_que.put(1)
                    while wait_que.qsize() < 2:
                        continue
                    wait_que.get()
                    stats, break_counter, best_pg_state, best_ee_state =\
                    func.checkpoint_func(args, program_generator, execution_engine, 
                                         train_loader, val_loader, t, epoch, stats,
                                         model_name, pg_loss, ee_loss, pg_kwargs, ee_kwargs, 
                                         vocab, break_counter, best_pg_state, best_ee_state)
                    pg_loss = []
                    ee_loss = []
                    wait_que.put(1)
                if break_counter >= args.break_counter:
                    cont = False
                    break


