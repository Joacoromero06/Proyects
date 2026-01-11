#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 08:15:14 2026

@author: joaco
"""
import torch
from torch import nn
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        """ Initializate trainable weight, Transfor d_in -> d_out"""
        super().__init__()
        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, X):
        """
        Compute K, Q, V correspond to input: X. By using application.
        Compute attention scores by encoding Sij = Qi . Kj.
        Normalize scores using softmax, scaling by d_out dimension.
        Compute context vector: Z. By weighting values via mat-mult.
        """
        # Matrix Aplication
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)
        
        # Encode of Dot Products
        S = Q @ K.T
        W = torch.softmax(S/(K.shape[-1] ** 0.5), dim=-1)
        
        # Weight of Values 
        Z = W @ V
        return Z
# %%
d_in, d_out = 3, 2
f = SelfAttention(d_in, d_out)
X = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
)


