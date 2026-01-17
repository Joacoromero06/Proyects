#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 21:01:53 2026

@author: joaco
"""

import urllib.request
url = (
"https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch05/"
"01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)

from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

#%%
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


# %%
print("params:", params.keys())
print("settings:", settings)

# Distinct configs depending on model version
model_configs = {
"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Scheme configuration for GPTModel class
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False   
    }

#%%

from GPT_Classes import GPTModel
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

# %%
from torch import nn
import torch
import numpy as np

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError("Mismatch error. Left {left.shape} ,Right {right.shape}. ")
    return nn.Parameter(torch.tensor(right))

# params: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
def load_weights2gpt(gpt:GPTModel, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split( 
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
            )
        gpt.trf_blocks[b].att.Wq.weight = assign(
        gpt.trf_blocks[b].att.Wq.weight, q_w.T)
        gpt.trf_blocks[b].att.Wk.weight = assign(
        gpt.trf_blocks[b].att.Wk.weight, k_w.T)
        gpt.trf_blocks[b].att.Wv.weight = assign(
        gpt.trf_blocks[b].att.Wv.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.Wq.bias = assign(
        gpt.trf_blocks[b].att.Wq.bias, q_b)
        gpt.trf_blocks[b].att.Wk.bias = assign(
        gpt.trf_blocks[b].att.Wk.bias, k_b)
        gpt.trf_blocks[b].att.Wv.bias = assign(
        gpt.trf_blocks[b].att.Wv.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
            )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
            )
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
            )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
            )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
            )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
            )
        
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
            )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
            )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = gpt.tok_emb.weight
# %%
torch.manual_seed(123)
device = torch.device("cpu")
load_weights2gpt(gpt, params)
gpt.to(device)
from Training_Generating import generate, text2IDs, IDs2text
import tiktoken; T = tiktoken.get_encoding("gpt2")
toks = generate(M=gpt, 
                idx=text2IDs("the sum of 2 + 2 is: ", T), 
                n_generate=25, 
                context_size=NEW_CONFIG["context_length"],
                top_k=50,
                temperature=0.4)
print(IDs2text(toks, T))
# %%   
# Check the first Attention Layer Query Weights
print("Model Wq mean:", gpt.trf_blocks[0].att.Wq.weight.mean().item())
print("Model Wq std: ", gpt.trf_blocks[0].att.Wq.weight.std().item())

# Compare with raw params (Need to manually split to compare)
q_w_raw, _, _ = np.split(params["blocks"][0]["attn"]["c_attn"]["w"], 3, axis=-1)
print("Param Wq mean:", q_w_raw.mean())
print("Param Wq std: ", q_w_raw.std())

# %%
# import loaders, variables defined in other file
from Lecture5_2 import train_loader, val_loader
from Training_Generating import train_model1, plot_losses

if __name__ == "__main__":
    OPT = torch.optim.AdamW(gpt.parameters(), lr=0.0004, weight_decay=0.1)
    T_losses, V_losses, toks_seen = train_model1(M=gpt, 
                                      T_loader=train_loader, 
                                      V_loader=val_loader, 
                                      OPT=OPT, 
                                      D=device, 
                                      n_epochs=10, 
                                      eval_freq=5, 
                                      n_eval_iter=5, 
                                      text_eg="Every effort moves you",
                                      T=T
                                      )
# %%
if __name__ == "__main__":
    epoch_tensor = torch.linspace(0,10,len(T_losses))
    plot_losses(epochs_seen=epoch_tensor, 
                tokens_seen=toks_seen, 
                T_losses=T_losses, 
                V_losses=V_losses)






