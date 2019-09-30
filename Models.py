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


class Seq2Seq(nn.module):
    def __init__(self,
                 encoder_vocab_size=100,
                 decoder_vocab_size=100,
                 wordvec_dim=300,
                 hidden_dim=256,
                 layers=2,
                 LSTM_dropout=0,
                 null_token=0,
                 start_token=1,
                 end_token=2,
                 encoder_embed=None):
        super(Seq2Seq, self).__init__()
        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_LSTM = nn.LSTM(wordvec_dim, hidden_dim, layers,
                                    dropout=LSTM_dropout, batch_first=True)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
        self.decoder_LSTM = nn.LSTM(wordvec_dim + hidden_dim, hidden_dim, layers,
                                    dropout=LSTM_dropout, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
        self.NULL = null_token
        self.START = start_token
        self.END = end_token
        
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
    
    def before_rnn:
        
        
    
    
