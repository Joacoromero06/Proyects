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
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False   
    }

import torch
from torch import nn
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

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(Self, X):
        return 0.5 * X * (1 + \
                    torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (X + 0.04715 * X**3) )
                    )

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

 
        
        
        