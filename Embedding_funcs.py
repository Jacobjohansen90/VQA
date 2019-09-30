#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:17:33 2019

@author: jacob
"""

def expand_embedding_vocab(embed, token_to_idx, word2vec=None, std=0.01):
    old_weight = embed.weight.data
    old_N, D = old_weight.size()
    new_N = 1 + max(idx for idx in token_to_idx.values())
    new_weight = old_weight.new(new_N, D).normal().mul_(std)
    new_weight[:old_N].copy_(old_weight)
    
    if word2vec is not None:
        assert D == word2vec['vecs'].size(1), 'Word vector dimension mismatch'
        word2vec_token_to_idx = {w: i for i, w in enumerate(word2vec['words'])}
        for token, idx in token_to_idx.items():
            word2vec_idx = word2vec_token_to_idx.get(token, None)
            if idx >= old_N and word2vec_idx is not None:
                vec = word2vec['vecs'][word2vec_idx]
                new_weight[idx].copy_(vec)
    
    embed.num_embeddings = new_N
    embed.weight.data = new_weight

    return embed