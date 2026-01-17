#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 21:43:26 2026

@author: joaco
"""

import torch
from torch import nn
torch.manual_seed(123)
X = torch.randn(2,5)
L = nn.Sequential(nn.Linear(5,6), nn.ReLU())
Y = L(X)

# Study mean and variance
u = Y.mean(dim=-1, keepdim=True)
o = Y.var(dim=-1, keepdim=True)
print(f"mean of Y={u}")
print(f"var of Y={o}")

# Normalize
Y_norm = (Y-u) / torch.sqrt(o)
u_norm = Y_norm.mean(dim=-1, keepdim=True)
o_norm = Y_norm.var(dim=-1, keepdim=True)
print(f"mean of Y normalized ={u_norm}")
print(f"var of Y normalize ={o_norm}")


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter( torch.ones(emb_dim) )
        self.shift = nn.Parameter( torch.zeros(emb_dim) )
    def forward(self, Y):
        u = Y.mean(dim=-1, keepdim=True)
        o = Y.var(dim=-1, keepdim=True, unbiased=False)
        Y_norm = (Y-u) / torch.sqrt(o + self.eps)
        return self.scale * Y_norm + self.shift

ln = LayerNorm(6)
ln(Y).mean(dim=-1)
ln(Y).var(dim=-1, unbiased=False)

# %%

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(Self, X):
        return 0.5 * X * (1 + \
                    torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (X + 0.04715 * X**3) )
                    )

import matplotlib.pyplot as plt
x = torch.linspace(-3, 3, 100)
relu, gelu = nn.ReLU(), GELU()
y_relu, y_gelu = relu(x), gelu(x)

plt.figure(figsize=(8,3))
for i, (y, label) in enumerate( zip([y_relu, y_gelu], ["RELU", "GELU"]), start=1 ):
    plt.subplot(1,2,i)
    plt.plot(x,y)
    plt.title(f'{label} activation function')
    plt.xlabel('x')
    plt.ylabel(f'{label}(x)')
    plt.grid(True)
    
# %%
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['emb_dim']
        self.layers = nn.Sequential(
            nn.Linear(d, 4*d), 
            GELU(),
            nn.Linear(4*d, d)
            )
    def forward(self, X):
        return self.layers(X)



