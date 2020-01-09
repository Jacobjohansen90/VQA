#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:31:25 2019

@author: jacob
"""

import os
from scipy.misc import imread, imresize
import torch

from Preprocess_funcs import build_model, run_batch

##Args

feature_model = 'resnet101'
split = 'val'
max_images = None
model_ = 'resnet101'
model_stage = 3
batch_size = 32
img_h = img_w = 224

##Preprocessing
if split == 'all':
    splits = ['test', 'train', 'val']
else:
    splits = [split]
for split in splits:
    output_dir = '../Data/images/'+split+'/'
    image_dir = '../Data/Dataset/images/'+split+'/'
    
    input_paths = []
    idx_set = set()
    
    for fn in os.listdir(image_dir):
        if not fn.endswith('.png'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        input_paths.append((os.path.join(image_dir, fn), idx))
        idx_set.add(idx)
    
    input_paths.sort(key=lambda x: x[1])
    assert len(idx_set) == len(input_paths)
    assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
    
    if max_images is not None:
        input_paths = input_paths[:max_images]
        
    print(input_paths[0])
    print(input_paths[-1])
    
    model = build_model(image_dir, output_dir, img_h, img_w, model_,
                        model_stage=model_stage, batch_size=batch_size)
    
    img_size = (img_h, img_w)
    feat_dset = None
    i0 = 0
    cur_batch = []
    paths = []
    
    for i, (path, idx) in enumerate(input_paths):
        img = imread(path, mode='RGB')
        img = imresize(img, img_size, interp='bicubic')
        img = img.transpose(2,0,1)[None]
        cur_batch.append(img)
        paths.append(path)
        if len(cur_batch) == batch_size:
            feats = run_batch(cur_batch, model)
            for j in range(feats.shape[0]):
                torch.save(feats[j], output_dir + paths[j].split('/')[-1])
            i1 = i0 + len(cur_batch)
            i0 = i1
            print('Processed %d / %d images' % (i1, len(input_paths)))
            cur_batch = []
            paths = []
    if len(cur_batch) > 0:
        feats = run_batch(cur_batch, model)
        for j in range(feats.shape[0]):
            torch.save(feats[j], output_dir + paths[j].split('/')[-1])
            #Files are saved with .png extension, slighty ambigious.
        i1 = i0 + len(cur_batch)
        print('Processed %d / %d images' % (i1, len(input_paths)))



