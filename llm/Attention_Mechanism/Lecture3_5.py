#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 09:17:07 2026

@author: joaco
"""

import torch
from torch import nn

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, n_context, dropout, qkv_bias=False):
        super().__init__()
        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.d_out= d_out
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(n_context, n_context), 
            diagonal=1) 
            )
    def forward(self, X):
        n_batch, n_token, d_in = X.shape
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        
        S = Q @ K.transpose(1,2) # transpose dimension n_token, d_in
        
        # Mask S with prev init mask, using current n° tokens.
        S.masked_fill_(self.mask.bool()[:n_token, :n_token], -torch.inf)        
        
        # Normalize W, rows sum up to 1
        W = torch.softmax(S/K.shape[-1] ** 0.5, dim=-1)
        
        # Randomly dropout W
        W = self.dropout(W)
        
        Z = W @ V
        return Z

# %%
import torch
torch.manual_seed(123)
ca = CasualAttention(d_in=3, d_out=2, n_context=6, dropout=0.5)
X = torch.tensor(
     [[0.43, 0.15, 0.89], # Your
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts
     [0.22, 0.58, 0.33], # with
     [0.77, 0.25, 0.10], # one
     [0.05, 0.80, 0.55]], # step
)
batch = torch.stack( (X, X), dim=0 )

    
    
