#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 07:53:06 2026

@author: joaco
"""

import os
abs_path = os.path.abspath(__file__)
cname = os.path.dirname(abs_path)
os.chdir(cname)

import torch
from torch import nn
from Lecture4_2 import FeedForward
from Lecture4_2 import LayerNorm
from Lecture3_6 import MultiHeadAttention
class TransformerBlock(nn.Module):
    """     Main Block in GPT architecture      """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in            = cfg["emb_dim"], 
            d_out           = cfg["emb_dim"], 
            context_length  = cfg["context_length"], 
            dropout         = cfg["drop_rate"], 
            n_heads         = cfg["n_heads"],
            qkv_bias        = cfg["qkv_bias"]
            )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        self.dropout = nn.Dropout(cfg["drop_rate"])
    def forward(self, X):
        shortcut = X
        X = self.norm1(X)
        X = self.att(X)
        X = self.dropout(X)
        X = X + shortcut
        
        shortcut = X
        X = self.norm2(X)
        X = self.ff(X)
        X = self.dropout(X)
        X = X + shortcut
        return X
  
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False   
    }
X = torch.randn(2,4,768)
block = TransformerBlock(GPT_CONFIG_124M)
Y = block(X)
