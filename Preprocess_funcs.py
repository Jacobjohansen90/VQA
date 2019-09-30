#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:10:30 2019

@author: jacob
"""

#Funcs for preprocess questions

"""
Utilities for preprocessing sequence data.
Special tokens included:

<NULL>: Extra parts of the sequence that we should ignore (padding)
<START>: Goes at the start of a sequence
<END>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
"""

special_tokens = {
        '<NULL>':0,
        '<START>':1,
        '<END>':2,
        '<UNK>':3}

def tokenize(s, delim=' ', add_start_token=True,
             add_end_token=True, punct_to_keep=None,
             punct_to_remove=None):
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))
        
    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    
    token_to_count = {}
    tokenize_kwargs = {
            'delim': delim,
            'punct_to_keep': punct_to_keep,
            'punct_to_remove': punct_to_remove}
    
    for seq in sequences:
        seq_tokens = tokenize(seq, **tokenize_kwargs, 
                              add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1
            
    token_to_idx = {}
    
    for token, idx in special_tokens.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)
            
    return token_to_idx

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx

def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
        if delim is None:
            return tokens
        else:
            return delim.join(tokens)
        
##Preprocess funcs for image
            
def build_model(img_dir, output_h5_file, img_h, img_w, model, model_stage=3,
                batch_size=64):
    if not hasattr(torchvision.models, model):
        raise ValueError('Invalid model "%s"' % model)
    if not 'resnet' in model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, model)(pretrained=True)
    layers = [cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool]
    for i in range(model_stage):
        name = 'layer%d' % (i+1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)
    model.cuda()
    model.eval()
    return model

def run_batch(cur_batch, model):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,3,1,1) #Comes from CLEVR
    std = np.array([0.229, 0.224, 0.224]).reshape(1,3,1,1) #Comes from CLEVR
    
    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    image_batch = torch.autograd.Variable(image_batch, volatile=True)
    
    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()
    
    return feats
    