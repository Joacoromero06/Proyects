#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 20:45:03 2026

@author: joaco
"""

import torch 

from GPT_Classes import generate_text1

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

def text2IDs(text, T):
    return torch.tensor(T.encode(text, allowed_special={'<|EOF|>'})).unsqueeze(0)
def IDs2text(ids, T):
    return T.decode(ids.squeeze(0).tolist())

def eval_model(M, T_loader, V_loader, D, n_eval_iter):
    M.eval()
    with torch.no_grad():
        T_loss = calc_loss_loader(T_loader, M, D, n_batches=n_eval_iter)
        V_loss = calc_loss_loader(V_loader, M, D, n_batches=n_eval_iter)
    M.train()
    return T_loss, V_loss
def gen_show_sample(M, T, D, text_eg):
    M.eval()
    # Row of pos embedding size, is the max context length or n° tokens M can handle
    context_size = M.pos_emb.weight.shape[0]
    
    in_ids = text2IDs(text_eg, T)
    with torch.no_grad():
        out_ids = generate_text1(M, in_ids, 
                        n_tokens_target=50, window_size=context_size)
    out_text = IDs2text(out_ids, T)
    print(out_text.replace("\n", " "))
    M.train()    
def train_model1(M, T_loader, V_loader, OPT, 
                 D, n_epochs, eval_freq, n_eval_iter, text_eg, T):
    """         Training model function-Training loop
    D: device
    eval_freq:
    eval_iter:
    text_eg: Text for run-time validation
    T: tokenizer
    """
    T_losses, V_losses, toks_seen = [], [], []
    acum_tokens, global_step = 0, -1 
    
    for epoch in range(n_epochs):
        M.train()
        for X, Y in T_loader:
            # Reset loss gradients from previous batch iteration
            OPT.zero_grad()
            
            # Compute loss Tensor for X: input, Y: target 
            loss = calc_loss_batch(X, Y, M, D)
            
            # Compute loss gradients  
            loss.backward()
            
            # Updates model weights
            OPT.step()
            
            acum_tokens += X.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                T_loss, V_loss = eval_model(M, T_loader, V_loader, D, n_eval_iter)
                T_losses.append(T_loss); V_losses.append(V_loss)
                toks_seen.append(acum_tokens)
                
                print(f"Ep {epoch+1} (step {global_step:06d}): "
                      f"Train loss {T_loss:.3f}, "
                      f"Val loss {V_loss:.3f}"
                     )
        gen_show_sample(M, T, D, text_eg)
    return T_losses, V_losses, toks_seen
            
# %%
"""Training GPTModel for 10 epochs using AdamW and tracking process"""

"""
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from GPT_Classes import GPT_CONFIG_124M, GPTModel
M = GPTModel(GPT_CONFIG_124M)
M.to(device)
OPT = torch.optim.AdamW(M.parameters(), lr=0.0004, weight_decay=0.1)
from Lecture5_2 import train_loader, val_loader
import tiktoken
T = tiktoken.get_encoding("gpt2")
T_losses, V_losses, toks_seen = train_model1(M, train_loader, val_loader, 
                                             OPT, D=device, n_epochs=10, 
                                             eval_freq=5, n_eval_iter=5, 
                                             text_eg="Every effort moves you", T=T)

torch.save({
     "model_state_dict": M.state_dict(),
     "optimizer_state_dict": OPT.state_dict()
     },
    "model_and_optimizer.pth"
    )
# %%
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, T_losses, V_losses):
    fig,ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, T_losses, label="Training loss")
    ax1.plot(epochs_seen, V_losses, label="Validation loss", linestyle="-.")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
"""
if "__main__" == __name__:
    ep_tensor = torch.linspace(0, 10, len(T_losses))
    plot_losses(ep_tensor, toks_seen, T_losses, V_losses)
"""
# %%


""" Text generation function """ 
def generate(M, idx, n_generate, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(n_generate):
        # from input we only see the last tokens
        window = idx[:, -context_size:]
        with torch.no_grad():
            Y = M(window)
        # Last element
        Y = Y[:, -1, :]

        if top_k is not None:
            # Get the top_k logits
            top_Y, _ = torch.topk(Y, top_k)
            min_top_Y = top_Y[:, -1]
            Y = torch.where(
                    Y < min_top_Y,
                    torch.tensor(float("-inf")).to(Y.device),
                    Y
                    )
        if temperature > 0.0:
            Y = Y / temperature
            probs = torch.softmax(Y, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(Y, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat( (idx, idx_next), dim=-1 )
    return idx
