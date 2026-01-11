#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 12:26:14 2026

@author: joaco
"""

import torch

# Representation for learning of the embedding-3dim for a small sequence
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
)
""" 
        Data Loader is an structure containing (inputs, targets)
inputs: tensor(rows: batch, column: context, depth: dimension)
"""

# Getting attention scores W[2][:] 
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, xi in enumerate(inputs):
   attn_scores_2[i] = torch.dot(xi, query)

# Sum approach
attn_weights_2_v1 = attn_scores_2 / attn_scores_2.sum()

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum()

# Not tested function
attn_weights_2_v2 = softmax_naive(attn_scores_2)

# PyTorch function
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    
"""
        Context vector Zj is calculated as sum combination of Xi*w[i][j]        
"""
Z2_aux = attn_weights_2 @ inputs

Z2 = torch.empty(query.shape)
for i, xi in enumerate(inputs):
    Z2 += xi * attn_weights_2[i]




    
    
    
    
    