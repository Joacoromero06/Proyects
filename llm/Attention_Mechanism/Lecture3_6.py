#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 21:55:00 2026

@author: joaco
"""

import os 
abs_path = os.path.abspath(__file__)
cname = os.path.dirname(abs_path)
os.chdir(cname)

import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout, n_heads, qkv_bias=False):
        super().__init__()

        
        # Setting of hyperparameters
        self.d_out = d_out
        self.n_heads = n_heads
        # Compute of dimension for each head
        self.head_dim = d_out // n_heads        
        
        # Initialization of weight matrices
        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Initialization of weight matrix for Z after concatanation
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Initialization of dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialization of mask matrix for masking attention weights 
        self.register_buffer('mask', 
                             torch.triu(torch.ones(context_length, context_length), diagonal=1) )        
    def forward(self, X):
        """             Forward: Contextualize input: X             """
        # Get dimension of the input, X(b,n_token,d_in)
        b, n_tokens, d_in = X.shape
        
        # Apply linear transformation to X, Q(b,n_token,d_out)
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        
        # Unroll last dim, change dimension adding Multi-Head
        Q, K, V = Q.view(b, n_tokens, self.n_heads, self.head_dim), K.view(b, n_tokens, self.n_heads, self.head_dim), V.view(b, n_tokens, self.n_heads, self.head_dim)
        
        # n_heads dimension works as batch dimension. Q(b, n_heads, n_token, head_dim)
        Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)
        
        # Compute attention scores, Sij = qi . kj. S(b, n_heads, n_token, n_token)
        S = Q @ K.transpose(2,3)
        
        # Mask attention scores, slicing mask with current n_token of input
        mask = self.mask.bool()[:n_tokens, :n_tokens]
        S.masked_fill_(mask, -torch.inf)
        
        # Compute attention weights
        W = torch.softmax(S / Q.shape[-1] ** 0.5, dim=-1)
        
        # Randomly dropout
        W = self.dropout(W)
        
        # Compute context tensor Z(b, n_heads, n_token, head_dim)
        Z = W @ V
        
        # Change dimension order, n_heads -> n_token to n_token -> n_heads
        Z = Z.transpose(1,2)
        
        # Flatten or concatenate Zi of each head into big_Z
        Z = Z.contiguous().view(b, n_tokens, self.d_out)
        
        # Apply linear projection to just concatanated Z
        Z = self.out_proj(Z)
        
        return Z
        
torch.manual_seed(123)
X = torch.tensor(
     [[0.43, 0.15, 0.89], # Your
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts
     [0.22, 0.58, 0.33], # with
     [0.77, 0.25, 0.10], # one
     [0.05, 0.80, 0.55]], # step
)
batch = torch.stack( (X,X), dim=0 )
b, n, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, n, 0.0, n_heads=2)


# %%   
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
[0.8993, 0.0390, 0.9268, 0.7388],
[0.7179, 0.7058, 0.9156, 0.4340]],
[[0.0772, 0.3565, 0.1479, 0.5331],
[0.4066, 0.2318, 0.4545, 0.9737],
[0.4606, 0.5159, 0.4220, 0.5786]]]])

fh = a[0,0,:,:]
sh = a[0,1,:,:]

# look is the same:
a@a.transpose(2,3)    
fh@fh.T
        
        
        
        
    