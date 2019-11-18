#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:27:41 2019

@author: jacob
"""

import json
import matplotlib.pyplot as plt

model = 'EE_10k'
show = True
path = '/home/jacob/Desktop/Specialle/Data/models/'+model+'.json'

d = open(path)
d = json.load(d)

best_val = d['best_val_acc']
best_t = d['best_model_t']

val_accs = d['val_accs']
train_accs = d['train_accs']

train_losses = d['train_losses']

ts = d['val_accs_ts']

plt.plot(ts, val_accs, 'tab:blue', label='Validation')
plt.plot(ts, train_accs, 'tab:orange', label='Train')
plt.title('Accuracy: %s \n Best val: %.4f at t: %d' %(model, best_val, best_t))
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.ylim([0.4,1])
plt.legend()
if show:
    plt.show()
else:
    plt.savefig('../Figures/'+model+'_accuracy.png')
    plt.close()

print(max(train_accs))
print(best_val)