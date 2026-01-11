#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:02:56 2026

@author: joaco
"""

import torch

# Representation for learning of the embedding-3dim for a small sequence
X = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
)

x2 = X[1]
d_in = X.shape[-1]
d_out = 2

# Matrix for save query, key, value vectors. R^D_inxD_out
torch.manual_seed(123)
Wq= torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
Wk= torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
Wv= torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# Tensor dim=2, rows: keys vectors. ki = xi @ Wk
K = X @ Wk
V = X @ Wv
Q = X @ Wq

# weight parameters for Self Attention: Wij = Qi dot kj; 
W22 = Q[1].dot(K[1])

# Wi = Qi @ K; Q e R^T X D_out; Qi e R^D_out;K e R^T X D_out
W2 = Q[1] @ K.T

# Q e R^TxD_out, K^T e R^D_outxT, Wij = Qi dot Kj, Wi = Qi @ K.T
W = Q @ K.T

# Scaling attention scores using embedding dimension of keys vectors
emb_dim_keys = K.shape[-1]
A = torch.softmax(W / emb_dim_keys ** 0.5, dim=-1)

# Context vector: Zi e R ^ D_out, Ai e R^T, V e R^TxD_out
Z2 = A[1] @ V





