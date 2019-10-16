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
                 max_length=30,
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
        self.max_length = max_length
        
    def expand_encoder_vocab(self, token_to_idx, word2vec=None, std=0.01):
        expand_embedding_vocab(self.encoder_embed, token_to_idx,
                               word2vec=word2vec, std=std)
        
    def get_dims(self, x=None, y=None):
        V_in = self.encoder_embed.num_embeddings
        V_out = self.decoder_embed.num_embeddings
        D = self.encoder_embed.embedding_dim
        H = self.encoder_LSTM.hidden_size
        L = self.encoder_LSTM.num_layers
        
        N = x.size(0) if x is not None else None
        N = y.size(0) if N is None and y is not None else N
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
    
    def logical_or(self, x, y):
        return (x+y).clamp_(0,1)
    
    def logical_not(self, x):
        return x == 0
    
    def encoder(self, x):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)
        embed = self.encoder_embed(x)
        h0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
        c0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
        
        out, _ = self.encoder_LSTM(embed, (h0, c0))
        
        #Gets the hidden state for the last non NULL value
        idx = idx.view(N, 1, 1).expand(N, 1, H)
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
            c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
        rnn_output, (ht, ct) = self.decoder_LSTM(rnn_input, (h0, c0))
        
        rnn_output_2d = rnn_output.contiguous().view(N*T_out, H)
        output_logprobs = self.decoder_linear(rnn_output_2d).view(N, T_out, V_out)
        
        return output_logprobs, ht, ct
    
    def reinforce_sample_MAPO(self, x, bloom_filter, temperature=1.0, 
                              argmax=True):
        N, T = x.size(0), self.max_length
        assert N == 1
        encoded = self.encoder(x)
        y = torch.LongTensor(N, T).fill_(self.NULL)
        cur_input = Variable(x.data.new(N,1).fill_(self.START))
        h, c = None, None
        multinomial_outputs = []
        multinomial_probs = []
        multinomial_m = []
        t = 0
        while t < T-1:
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            logprobs = logprobs / temperature
            probs = F.softmax(logprobs.view(N, -1), dim=1)
            m = torch.distributions.Categorical(probs)
            if argmax:
                _, cur_input = probs.max(1)
            else:
                cur_output = m.sample()
            y[:, t] = cur_output.data.cpu()
            prg_tmp = '-'.join(str(e) for e in y[0,:t+1].tolist())
            while bloom_filter.check(prg_tmp) and sum(probs) > 0:
                probs[cur_output] = 0
                m = torch.distributions.Categorical(probs)
                if argmax:
                    _, cur_output = probs.max(1) 
                else:
                    cur_output = m.sample()
                y[:, t] = cur_output.data.cpu()
                prg_tmp = '-'.join(str(e) for e in y[0,:t+1].tolist())
            if bloom_filter.check(prg_tmp) == 0:
                t += 1
                cur_input = cur_output.unsqueeze(1)
            else:
                prg_tmp = '-'.join(str(e) for e in y[0,:t].tolist())
                bloom_filter.add(prg_tmp)
                break
            if cur_output.data.cpu() == self.END:
                bloom_filter.add(prg_tmp)
                break
            multinomial_outputs.append(cur_output)
            multinomial_probs.append(probs)
            multinomial_m.append(m)
        return Variable(y.type_as(x.data)), bloom_filter, multinomial_outputs, \
               multinomial_probs, multinomial_m
                
    def reinforce_sample(self, x, temperature=1.0, argmax=False):
        N, T = x.size(0), self.max_length
        encoded = self.encoder(x)
        y = torch.LongTensor(N, T).fill_(self.NULL)
        done = torch.ByteTensor(N).fill_(0)
        cur_input = Variable(x.data.new(N,1).fill_(self.START))
        h, c = None, None
        self.multinomial_outputs = []
        self.multinomial_probs = []
        self.multinomial_m = []
        for t in range(T):
            #The logprobs are N x 1 x V
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            logprobs = logprobs / temperature
            probs = F.softmax(logprobs.view(N, -1), dim=1) #Makes logprobs N x V
            m = torch.distributions.Categorical(probs)
            if argmax:
                _, cur_output = probs.max(1)
            else:
                cur_output = m.sample() #Now N
            self.multinomial_outputs.append(cur_output)
            self.multinomial_probs.append(probs)
            self.multinomial_m.append(m)
            cur_output_data = cur_output.data.cpu()
            not_done = self.logical_not(done)
            y[:, t][not_done] = cur_output_data[not_done]
            done = self.logical_or(done, (cur_output_data.cpu()==self.END).type(torch.uint8))
            cur_input = cur_output.unsqueeze(1)
            if done.sum() == N:
                break
        return Variable(y.type_as(x.data))
    
    def compute_loss(self, output_logprobs, y):
        """
        Assumes first element of y is start token.
        Assumes end of y is padded with null tokens until T_out length
        """
        self.multinomial_outputs = None
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
        mask = y.data != self.NULL
        y_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
        y_mask[:, 1:] = mask[:, 1:]
        y_masked = y[y_mask]
        out_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
        out_mask[:,:-1] = mask[:,1:]
        out_mask = out_mask.view(N, T_out, 1).expand(N, T_out, V_out)
        out_masked = output_logprobs[out_mask].view(-1, V_out)
        loss = F.cross_entropy(out_masked, y_masked)
        return loss
    
    def forward(self, x, y):
        encoded = self.encoder(x)
        output_logprobs, _, _ = self.decoder(encoded, y)
        loss = self.compute_loss(output_logprobs, y)
        return loss
    
    def reinforce_backward(self, reward, output_mask=None):
        assert self.multinomial_outputs is not None, 'Must call reinforce sample first'
        
        def gen_hook(mask):
            def hook(grad):
                return grad * mask.contiguous().view(-1,1).expand_as(grad)
            return hook
        
        if output_mask is not None:
            for t, probs in enumerate(self.multinomial_probs):
                mask = Variable(output_mask[:, t])
                probs.register_hook(gen_hook(mask))
        
        for i in range(len(self.multinomial_outputs)):
            sampled_output = self.multinomial_outputs[i]
            m = self.multinomial_m[i]
            loss = -(m.log_prob(sampled_output)*reward).sum()
            loss.backward(retain_graph=True)
            
    def reinforce_backward_MAPO(self, multinomial_outputs, multinomial_m,
                                multinomial_probs, reward, output_mask=None):
        assert multinomial_outputs is not None, 'Must call reinforce sample MAPO first'
        def gen_hook(mask):
            def hook(grad):
                return grad * mask.contiguous().view(-1,1).expand_as(grad)
            return hook
        
        if output_mask is not None:
            for t, probs in enumerate(multinomial_probs):
                mask = Variable(output_mask[:,t])
                probs.register_hook(gen_hook(mask))
                
        for i in range(len(multinomial_outputs)):
            sampled_output = multinomial_outputs[i]
            m = multinomial_m[i]
            loss = -(m.log_prob(sampled_output)*reward).sum()
            loss.backward(retain_graph=True)    
            
    def sample(self, x, max_length=50):
        self.multinomial_outputs = None
        encoded = self.encoder(x)
        y = [self.START]
        h0, c0 = None, None
        while True:
            cur_y = Variable(torch.LongTensor([y[-1]]).type_as(x.data).view(1,1))
            logprobs, h0, c0 = self.decoder(encoded, cur_y, h0, c0)
            _, next_y = logprobs.data.max(2)
            y.append(next_y.item())
            if len(y) >= max_length or y[-1] == self.END:
                break
        return y

            
        
    
    
