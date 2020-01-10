#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:27:24 2019

@author: jacob
"""
import Functions as func
import torch
import argparse
from DataLoader import ClevrDataLoader
import torch.multiprocessing as mp
from MAPO_workers import MAPO_CPU
#TODO: image idx start from / mask does not work in dataloader
#TODO: RL rewards are not recorded

#%% Setup Params
if __name__ == '__main__':
    mp.set_start_method('forkserver')

    parser = argparse.ArgumentParser()

    parser.add_argument('--multi_GPU', default=True) #Use all avalaible GPUs?
    parser.add_argument('--info', default=False)
    #Do you want all info or minimal?

    #Training length / early stopping
    parser.add_argument('--num_iterations', default=200000, type=int)
    parser.add_argument('--break_after', default=5, type=int)
    #If val has not improved after break_after checks, we early stop

    #Samples and shuffeling
    parser.add_argument('--shuffle_train_data', default=True, type=int)
    parser.add_argument('--num_PG_samples', default=1000, type=int)
    parser.add_argument('--PG_num_of_each', default=20, type=int) #No larger than 24
    parser.add_argument('--oversample', default=True)
    parser.add_argument('--num_train_samples', default=None, type=int) 
    parser.add_argument('--num_val_samples', default=None, type=int)
    #If None we load all examples

    # Optimization options
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate_PG', default=5e-4, type=float)
    parser.add_argument('--learning_rate_EE', default=1e-4, type=float)
    parser.add_argument('--learning_rate_MAPO', default=5e-5, type=float)
    parser.add_argument('--L2_pg', default=0.001, type=float)
    parser.add_argument('--L2_ee', default=0.0001, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    
    # Output options
    parser.add_argument('--checkpoint_every', default=1000, type=int)

    # What type of model to use and which parts to train
    parser.add_argument('--model_type', default='all')

    # Start from an existing checkpoint
    parser.add_argument('--pg_start_from', default=None)
    parser.add_argument('--ee_start_from', default=None)

    #Bloom Filter options
    parser.add_argument('--bf_est_ele', default=10**3, type=int)
    parser.add_argument('--bf_false_pos_rate', default=0.01, type=float)
    
    #MAPO
    parser.add_argument('--MAPO_max_cpus', default=24, type=int)
    parser.add_argument('--MAPO_qsize', default=320, type=int)
    parser.add_argument('--MAPO_sample_argmax', default=False)
    parser.add_argument('--MAPO_check_bf_argmax', default=False)
    parser.add_argument('--MAPO_programs_pr_pass', default=20, type=int)
    parser.add_argument('--pg_RL', default=True) #Train PG negatively using RL
    parser.add_argument('--alpha', default=1, type=float) #Weight for RL loss
    
    #Datapaths
    parser.add_argument('--train_questions', default='../Data/questions/train.npy')
    parser.add_argument('--train_features', default='../Data/images/train/')
    parser.add_argument('--val_questions', default='../Data/questions/val.npy')
    parser.add_argument('--val_features', default='../Data/images/val/')
    parser.add_argument('--vocab_json', default='../Data/vocab/vocab.json') 
    parser.add_argument('--high_reward_path', default='../Data/high_reward_paths/')    
    parser.add_argument('--checkpoint_path', default='../Data/models/')
    parser.add_argument('--bf_load_path', default='../Data/bloom_filters/')
    parser.add_argument('--path_to_index_file', default='../Data/questions/ans_to_index.npy')
    
    #Dataloader params
    parser.add_argument('--feature_dim', default='1024,14,14')
    parser.add_argument('--loader_num_workers', type=int, default=3)
    
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
    checkpoint_path = None
    
    val_loader_kwargs = {
            'question_path': args.val_questions,
            'feature_path': args.val_features,
            'vocab': vocab,
            'batch_size': args.batch_size,
            'max_samples': args.num_val_samples,
            'num_workers': args.loader_num_workers}

    val_loader = ClevrDataLoader(**val_loader_kwargs)
    
    train_acc_kwargs = {
            'question_path': args.train_questions,
            'feature_path': args.train_features,
            'vocab': vocab,
            'batch_size': args.batch_size,
            'max_samples': args.num_train_samples,
            'num_workers': args.loader_num_workers}
    
    train_acc_loader = ClevrDataLoader(**train_acc_kwargs)
    
    if args.model_type == 'all':
        model = ['PG', 'EE', 'MAPO']
    else:
        model = args.model_type.split('+')
    
    for model_ in model:
        print('Training %s' % model_)
        #Setup current model        
        program_generator, pg_kwargs, pg_optimizer = None, None, None
        execution_engine, ee_kwargs, ee_optimizer = None, None, None
        
        best_pg_state, best_ee_state = None, None
        
        #Load previous trained models if needed
        if args.model_type == 'all':        
            if model_ == 'EE' and args.pg_start_from is None:
                args.pg_start_from = checkpoint_path + '.pt'
            if model_ == 'MAPO' and args.ee_start_from is None:
                args.ee_start_from = checkpoint_path + '.pt'


        if model_ == 'PG':
            balanced_n = args.PG_num_of_each
            oversample = False
            program_generator, pg_kwargs = func.get_program_generator(vocab, args)        
            pg_optimizer = torch.optim.Adam(program_generator.parameters(),
                                            lr=args.learning_rate_PG,
                                            weight_decay=args.L2_pg)

        elif model_ == 'EE':
            balanced_n = None
            oversample = args.oversample
            program_generator, pg_kwargs = func.get_program_generator(vocab, args)        
            execution_engine, ee_kwargs = func.get_execution_engine(vocab, args)
            ee_optimizer = torch.optim.Adam(execution_engine.parameters(),
                                                lr=args.learning_rate_EE,
                                                weight_decay=args.L2_ee)
        elif model_ == 'MAPO':
            balanced_n = None
            oversample = args.oversample
            program_generator, pg_kwargs = func.get_program_generator(vocab, args)
            execution_engine, ee_kwargs = func.get_execution_engine(vocab, args)
            pg_optimizer = torch.optim.Adam(program_generator.parameters(),
                                            lr=args.learning_rate_MAPO,
                                            weight_decay=args.L2_pg)
            ee_optimizer = torch.optim.Adam(execution_engine.parameters(),
                                            lr=args.learning_rate_MAPO,
                                            weight_decay=args.L2_ee)
            
        #Auto checkpointing        
        model_name = func.auto_namer(model_, args)
        checkpoint_path = args.checkpoint_path + model_name
        
        #Setup loss and train loader
        loss_fn = torch.nn.CrossEntropyLoss().cuda()      
        
        if model_ == 'PG':
            max_samples = args.num_PG_samples
        else:
            max_samples = args.num_train_samples

                
        train_loader_kwargs = {
            'question_path': args.train_questions,
            'feature_path': args.train_features,
            'vocab':vocab,
            'batch_size':args.batch_size,
            'shuffle': args.shuffle_train_data,
            'pin_memory': True,
            'max_samples': max_samples,
            'num_workers': args.loader_num_workers,
            'balanced_n':balanced_n,
            'oversample':oversample,
            'path_to_index': args.path_to_index_file}
        
        train_loader = ClevrDataLoader(**train_loader_kwargs)    
         
        if model_ == 'MAPO':
                                 
            #Spawn MAPO workers and dataloader    
            cpu_count = mp.cpu_count()
            execution_engine.share_memory()
            program_generator.share_memory()
            
            if args.MAPO_max_cpus is not None:
                cpu_count = min(cpu_count, args.MAPO_max_cpus)
            if args.info:
                print('MAPO will use %d CPUs' % cpu_count)
            processes = []
            change_que = mp.Queue()
            sample_que = mp.Queue()
            
            for cpu in range(cpu_count-args.loader_num_workers-1):
                p = mp.Process(target=MAPO_CPU, args=(args, program_generator.cpu(),  
                                                      sample_que, cpu))
         
                p.start() 
                processes.append(p)
                if args.info:
                    print('MAPO worker %s spawned' % str(cpu))
            #Set model to GPU            
            program_generator.cuda()
            execution_engine.cuda()

            #Fill high reward buffer
            func.clean_up(args)
            func.set_mode('eval', [program_generator, execution_engine])      
            print('Making HR paths')
            hr_list = func.make_HR_paths(args, program_generator, execution_engine, train_loader)            
            p = mp.Process(target=func.MAPO_loader, args=(args, hr_list, change_que, 
                                                          sample_que, vocab, 
                                                          train_loader.dataset.sample_list))
            p.start()
            processes.append(p)            
            if args.info:
                print('Clevr dataloader spawned')
            
            #Keep track of novel path tries
            novel_counter = torch.zeros(len(train_loader)*args.batch_size)
        stats = {'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
                 'train_accs':[], 'val_accs': [], 'val_accs_ts': [],
                 'best_val_acc': -1, 'best_model_t': 0, 'epoch': []}
        
        t, epoch, reward_moving_avg = 0,0,0
        pg_loss = []
        ee_loss = []
        break_counter = 0
        cont = True 
        
        func.set_mode('train', [program_generator, execution_engine])
        
        while cont:
            inner_cont = True
            if model_ == 'PG':
                while inner_cont:
                    for batch in train_loader:
                        t += 1
                        questions, _, feats, answers, programs, _, _, _, _, _ = batch
                        pg_optimizer.zero_grad()
                        loss = program_generator(questions.cuda(), programs.cuda()).mean()
                        #mean is needed for multi GPU, has no impact if 1 GPU
                        loss.backward() 
                        pg_loss.append(loss.item())
                        pg_optimizer.step()
                       
                        if t % args.checkpoint_every == 0:
                            stats, break_counter, best_pg_state, best_ee_state =\
                            func.checkpoint_func(args, model_, program_generator, execution_engine, 
                                                 train_acc_loader, val_loader, t, epoch, stats,
                                                 model_name, pg_loss, ee_loss, pg_kwargs, ee_kwargs, 
                                                 vocab, break_counter, best_pg_state, best_ee_state,
                                                 checkpoint_path)
                            pg_loss = []
                            if break_counter >= args.break_after:
                                cont = False
                                inner_cont = False
                                break
                        
            elif model_ == 'EE':
                while inner_cont:
                    for batch in train_loader:
                        t += 1
                        questions, _, feats, answers, _, _, _, _, _, _ = batch
                        if args.multi_GPU:
                            programs_pred = program_generator.module.reinforce_sample(questions.cuda())
                        else:
                            programs_pred = program_generator.reinforce_sample(questions.cuda())
                        scores = execution_engine(feats.cuda(), programs_pred.cuda())
                        ee_optimizer.zero_grad()
                        loss = loss_fn(scores, answers.cuda())
                        loss.backward()
                        ee_loss.append(loss.item())
                        ee_optimizer.step()
    
                        if t % args.checkpoint_every == 0:
                            stats, break_counter, best_pg_state, best_ee_state =\
                            func.checkpoint_func(args, model_, program_generator, execution_engine, 
                                                 train_acc_loader, val_loader, t, epoch, stats,
                                                 model_name, pg_loss, ee_loss, pg_kwargs, ee_kwargs, 
                                                 vocab, break_counter, best_pg_state, best_ee_state, 
                                                 checkpoint_path)
                            ee_loss = []
                            
                            if break_counter >= args.break_after:
                                cont = False
                                inner_cont = False
                                break

            elif model_ == 'MAPO':
                while inner_cont:
                    for batch in train_loader:
                        t += 1
                        questions, _, feats, answers, _, _, indexs, _, programs, I = batch
                        I = I.squeeze()
                        #Test MAPO suggestions
                        func.set_mode('eval', [execution_engine])
                        for i in range(len(programs)):
                            if programs[i].shape == 1 and programs[i] != programs[i]:
                                continue
                            else:
                                scores = execution_engine(feats[i].unsqueeze(0).expand(len(programs[i]),1024,14,14).cuda(),
                                                          programs.cuda())
                                _, preds = scores.data.cpu.max(1)
                                I_test = (preds == answers[i].expand(len(programs[i])))
                                if I_test.sum() > 0:
                                    change_index = indexs[i]
                                    change_programs = programs[i][I_test]
                                    change_que.put((change_index, change_programs, 'positive'))
                                    I[i] = True
                                    programs[i] = programs[i][I_test][0] 
                                    #0 is most likely program sequence ,-1 is least
                        func.set_mode('train', [execution_engine])
                        
                        #Force programs if no high reward path
                        if args.multi_GPU and torch.cuda.device_count() > 1:
                            programs[~I] = program_generator.module.reinforce_sample(questions[~I].cuda())
                        else:
                            programs[~I] = program_generator.reinforce_sample(questions[~I].cuda())
                        
                        scores = execution_engine(feats.cuda(), programs.cuda())
                        _, preds = scores.data.cpu().max(1)

                        #Train EE with positive and negative examples
                        ee_optimizer.zero_grad()
                        loss = loss_fn(scores, answers.cuda())
                        loss.backward()
                        ee_loss.append(loss.item())
                        ee_optimizer.step()
                        #Check that all examples are still the same as originally (posistive and negative)
                        I_ = (preds==answers)
                        if I_ != I:
                            #These indexes have become negative
                            if ((I != I_) == I).sum() != 0:
                                change_indexs = indexs[(I != I_) == I] 
                                change_programs = programs[(I != I_) == I]
                                change_que.put((change_indexs, change_programs, 'negative'))
                                I[(I != I_) == I] = False

                            if (I[I_ == True] != True).sum() != 0:
                                #These indexes have become positive
                                change_indexs = indexs[(I_ == True)][I[I_ == True] != True]
                                change_programs = programs[(I_ == True)][I[I_ == True] != True]
                                change_que.put((change_indexs, change_programs, 'positive'))
                                I[I_ == True] = True

                        #PG positive examples training using backprop
                        pg_optimizer.zero_grad()
                        loss = program_generator(questions[I].cuda(), programs[I].cuda()).mean()
                        loss.backward()
                        pg_loss.append(loss.item())
                        pg_optimizer.step()
                        
                        #PG negative examples training using RL
                        if args.pg_RL:
                            raw_reward = (preds == answers).float()
                            reward_moving_avg *= args.reward_decay
                            reward_moving_avg += (1.0 - args.reward_decay) * raw_reward.mean()
                            centered_reward = raw_reward - reward_moving_avg
                                                    
                            pg_optimizer.zero_grad()
                            program_generator.reinforce_backward(centered_reward.cuda(), args.alpha)
                            pg_optimizer.step()
                        
                                                                                  
                        if t % args.checkpoint_every == 0:
                            stats, break_counter, best_pg_state, best_ee_state =\
                            func.checkpoint_func(args, model_, program_generator, execution_engine, 
                                                 train_acc_loader, val_loader, t, epoch, stats,
                                                 model_name, pg_loss, ee_loss, pg_kwargs, ee_kwargs, 
                                                 vocab, break_counter, best_pg_state, best_ee_state,
                                                 checkpoint_path)
                            pg_loss = []
                            ee_loss = []
                            if break_counter >= args.break_after:
                                for p in processes:
                                    p.terminate()
                                cont = False
                                inner_cont = False
                                break
    print('All models are done training')
