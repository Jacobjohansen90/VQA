#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:50:22 2019

@author: jacob
"""

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Embedding_funcs import expand_embedding_vocab


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder_vocab_size=100,
                 decoder_vocab_size=100,
                 wordvec_dim=300,
                 hidden_dim=256,
                 rnn_num_layers=2,
                 rnn_dropout=0,
                 null_token=0,
                 start_token=1,
                 end_token=2,
                 encoder_embed=None):
        super(Seq2Seq, self).__init__()
        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_LSTM = nn.LSTM(wordvec_dim, hidden_dim, rnn_num_layers,
                                    dropout=rnn_dropout, batch_first=True)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
        self.decoder_LSTM = nn.LSTM(wordvec_dim + hidden_dim, hidden_dim, 
                                    rnn_num_layers, dropout=rnn_dropout, 
                                    batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
        self.NULL = null_token
        self.START = start_token
        self.END = end_token
        self.multinomial_outputs = None
        
    def expand_encoder_vocab(self, token_to_idx, word2vec=None, std=0.01):
        expand_embedding_vocab(self.encoder_embed, token_to_idx,
                               word2vec=word2vec, std=std)
        
    def get_dims(self, x=None, y=None):
        V_in = self.encoder_embed.num_embeddings
        V_out = self.decocer_embed.num_embeddings
        D = self.encoder_embed.embedding_dim
        H = self.encoder_LSTM.hidden_size
        L = self.encoder_LSTM.num_layers
        
        N = x.size(0) if x is not None else None
        N = y.size(0) if N is not None and y is not None else N
        T_in = x.size(1) if x is not None else None
        T_out = y.size(1) if y is not None else None
        
        return V_in, V_out, D, H, L, N, T_in, T_out
    
    def before_rnn(self, x, replace=0):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T-1)
        
        x_cpu = x.cpu()
        for i in range(N):
            for t in range(T-1):
                if x_cpu.data[i, t] != self.NULL and x_cpu.data[i, t+1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data)
        x[x.data == self.NULL] = replace
        return x, Variable(idx)
    
    def logical_or(x, y):
        return (x+y).clamp_(0,1)
    
    def logical_not(x):
        return x == 0
    
    def encoder(self, x):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)
        embed = self.encoder_embed(x)
        h0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
        c0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
        
        out, _ = self.encoder_LSTM(embed, (h0, c0))
        
        #Gets the hidden state for the last non NULL value
        idx = idx.view(N, 1, 1).expand(H, 1, H)
        return out.gather(1, idx).view(N, H)
    
    def decoder(self, encoded, y, h0=None, c0=None):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
        
        if T_out > 1:
            y, _ = self.before_rnn(y)
        y_embed = self.decoder_embed(y)
        encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
        rnn_input = torch.cat([encoded_repeat, y_embed], 2)
        if h0 is None:
            h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
        if c0 is None:
            c0 = Variable(torch.zeros(L, N, H).type_as(encoded.dat))
        rnn_output, (ht, ct) = self.decoder_LSTM(rnn_input, (h0, c0))
        
        rnn_output_2d = rnn_output.contiguous().view(N*T_out, H)
        output_logprobs = self.decoder_linear(rnn_output_2d).view(N, T_out, V_out)
        
        return output_logprobs, ht, ct
    
    def reinforce_sample(self, x, max_length=30, temperature=1.0, argmax=False):
        N, T = x.size(0), max_length
        encoded = self.encoder(x)
        y = torch.LongTensor(N, T).fill_(self.NULL)
        done = torch.ByteTensor(N).fill_(0)
        cur_input = Variable(x.data.new(N,1).fill_(self.START))
        h, c = None, None
        self.multinomial_outputs = []
        self.multinomial_probs = []
        for t in range(T):
            #The logprobs are N x 1 x V
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            logprobs = logprobs / temperature
            probs = F.softmax(logprobs.view(N, -1)) #Makes logprobs N x V
            if argmax:
                _, cur_output = probs.max(1)
            else:
                cur_output = probs.multinomial() #Now N x 1
            self.multinomial_outputs.append(cur_output)
            self.multinomial_probs.append(probs)
            cur_output_data = cur_output.data.cpu()
            not_done = self.logical_not(done)
            y[:, t][not_done] = cur_output_data[not_done]
            done = self.logical_or(done, cur_output_data.cpu()==self.END)
            cur_input = cur_output
            if done.sum() == N:
                break
        return Variable(y.type_as(x.data))
    

            
        
    
    
