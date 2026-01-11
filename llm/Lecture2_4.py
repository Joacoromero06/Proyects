#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 08:50:31 2026

@author: joaco
"""

import os 
abs_path = os.path.abspath(__file__)
cname = os.path.dirname(abs_path)
os.chdir(cname)

import torch 
torch.manual_seed(123)

# %% 

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
from Lecture2_3 import create_dataloader_v1
n_context = 4
dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=n_context,
        stride=n_context,
        shuffle=False        
    ) 

# %%
n_vocab, n_dim = 50257, 256
emb_lay = torch.nn.Embedding(n_vocab, n_dim)

iter_data = iter(dataloader)
inputs, targets = next(iter_data)

# Tokens-Embedding: tensor(batch_size X max_length X n_dim)
token_emb = emb_lay(inputs)

# Pos-Embedding: tensor(max_legth X n_dim)
pos_emb_layer = torch.nn.Embedding(n_context, n_dim)
pos_emb = pos_emb_layer(torch.arange(n_context))

input_embedding = token_emb + pos_emb




