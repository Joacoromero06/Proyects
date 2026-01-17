#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 08:01:58 2026

@author: joaco
"""

import torch
from torch import nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
                *[ DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"]) ])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        b, seq_len = in_idx.shape
        tok_embs = self.tok_emb(in_idx)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) 
        
        X = tok_embs + pos_embs # Tensor (b,n,dim)
        X = self.drop_emb(X)
        X = self.trf_blocks(X)
        X = self.final_norm(X)
        logits = self.out_head(X)
        return logits
        
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    def forward(self, X):
        return X
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, X):
        return X
    
# %%


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = [] 
batch.append(torch.tensor(tokenizer.encode("Every effort moves you")))
batch.append(torch.tensor(tokenizer.encode("You are responsible of")))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False   
    }

model = DummyGPTModel(GPT_CONFIG_124M)
model(batch)

