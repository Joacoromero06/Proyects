#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 21:33:50 2026

@author: joaco
"""
import os, sys
abs_path = os.path.abspath(__file__)
cname = os.path.dirname(abs_path)
os.chdir(cname)

# %%

# 1. Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory (go up one level)
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent directory to sys.path
sys.path.append(parent_dir)



#%%

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

import tiktoken
T = tiktoken.get_encoding("gpt2")


train_ratio = 0.9
split_idx = int(train_ratio*len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

import torch
torch.manual_seed(123)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False   
    }
from Tokenization_Embedding.Lecture2_3 import create_dataloader_v1

train_loader = create_dataloader_v1(
    text=train_data,
    batch_size=2,
    stride=GPT_CONFIG_124M["context_length"],
    max_length=GPT_CONFIG_124M["context_length"],
    shuffle=True,
    drop_last=True,
    num_workers=0
    )
val_loader = create_dataloader_v1(
    text=val_data,
    batch_size=2,
    stride=GPT_CONFIG_124M["context_length"],
    max_length=GPT_CONFIG_124M["context_length"],
    shuffle=False,
    drop_last=False,
    num_workers=0
    )
if __name__ == __file__:
    for x,y in train_loader:
        print(x.shape, y.shape)
    print("====="*5)
    for x,y in val_loader:
        print(x.shape, y.shape)
    

# %% 
def calc_loss_batch(X, Y, model, D):
    """
    X: input batch of IDs(b,n)
    Y: target batch of IDs (b,n)
    D: device
    """
    X, Y = X.to(D), Y.to(D)
    
    logits = model(X)
    # Join batch dim to n° token dim
    loss = torch.nn.functional.cross_entropy(
        input=logits.flatten(0,1), target=Y.flatten() )
    
    return loss

def calc_loss_loader(data_loader, model, device, n_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif n_batches == None:
        n_batches = len(data_loader)
    else:
        n_batches = min(len(data_loader), n_batches)
    for i, (X,Y) in enumerate(data_loader):
        if i < n_batches:
            loss = calc_loss_batch(X, Y, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / n_batches

if __name__ == __file__:
    from GPT_Classes import GPTModel
    model = GPTModel(GPT_CONFIG_124M)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model)
        val_loss = calc_loss_loader(val_loader, model)
    
    
    
    
    
    
    
    
    
    
