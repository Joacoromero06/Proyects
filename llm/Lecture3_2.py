#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 14:46:10 2026

@author: joaco
"""

"""     Compute Attention weights for all input tokens      """

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
W_aux = torch.empty(6,6)
for i, xi in enumerate(X):
    for j, xj in enumerate(X):
        W_aux[i,j] = torch.dot(xi,xj)

# Matrix of weights, Wij = score of Xi: query , Xj: input
W = X @ X.T
# Matrix normalized
A = torch.softmax(W, dim=-1)
# Context vecs, Zi: context vec -> Xi. Zi = Wi @ X, W e R^TxT, X e R^TxD
Z = A @ X
