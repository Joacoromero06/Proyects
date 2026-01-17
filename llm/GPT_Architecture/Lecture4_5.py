#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 06:47:45 2026

@author: joaco
"""
import os 
abs_path = os.path.abspath(__file__)
cname = os.path.dirname(abs_path)
os.chdir(cname)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False   
    }
import torch
from torch import nn
from Lecture4_4 import TransformerBlock
from Lecture4_2 import LayerNorm
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
 
        
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) 
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
            )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, X_ids):
        b,n = X_ids.shape
        
        tok_embs = self.tok_emb(X_ids)
        pos_embs = self.pos_emb(torch.arange(n, device=X_ids.device))
        X = tok_embs + pos_embs
        
        X = self.drop_emb(X)
        X = self.trf_blocks(X)
        X = self.final_norm(X)
        Y = self.out_head(X)
        return Y

model = GPTModel(GPT_CONFIG_124M)
Y = model(torch.tensor([
                    [2,5,8,1],
                    [1,3,6,9]
                    ])
    )
total_params = sum (p.numel() for p in model.parameters())
        
# First look to weight tying GPT2 perform
#print('shape of token embedding layer: ', model.tok_emb.weight.shape)
#print('shape of output layer:          ',model.out_head.weight.shape)        

params_gpt2 = total_params - sum(p.numel() for p in model.tok_emb.parameters() )
  

# compute memory necessary for model
n_bytes = 4 * total_params 
m_mb = n_bytes / (1024 * 1024)
        
def generate_text1(model, idx, n_tokens_target, window_size):
    """ Simple generate text function
    model: GPTModel
    idx: id's tensor (b,n) of the input
        b: n°sentences
        n: current n° tokens in sentence
    n_tokens_target: n° tokens to generate
    window_size: context legth size delimiter the model handles
    """
    for _ in range(n_tokens_target):
        window = idx[:, -window_size:]
        with torch.no_grad():
            Y = model(window)
        
        # Last token of the prediction Y(b,n,50257)
        Y = Y[:, -1, :]
        prob = torch.softmax(Y, dim=-1)
        
        # Compute next token for each sentence
        # Tensor indice and token id is the same in nn.Embedding
        next_tokens = torch.argmax(prob, dim=-1, keepdim=True) 
        
        # Append next_token to input of id tokens
        idx = torch.cat((idx, next_tokens), dim=-1)
    return idx


import tiktoken
T = tiktoken.get_encoding("gpt2")        
ids = torch.tensor(T.encode("Hello, I am")).unsqueeze(0)
out = generate_text1(model, ids , 
              n_tokens_target=6, 
             window_size=GPT_CONFIG_124M["context_length"])
result = T.decode(out.squeeze().tolist())                   
        
        
        
        