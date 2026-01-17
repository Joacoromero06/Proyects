#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 07:39:37 2026

@author: joaco
"""
from Lecture5_2 import calc_loss_batch, calc_loss_loader
from Lecture5_1 import text2IDs, IDs2text
from GPT_Classes import generate_text1
import torch
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

# %%
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
        
if "__main__" == __name__:
    ep_tensor = torch.linspace(0, 10, len(T_losses))
    plot_losses(ep_tensor, toks_seen, T_losses, V_losses)

# %%

"""Decoding Strategies to control Randomness"""

# Plug trained model to generete_text1()

toks = generate_text1(model=M, 
                      idx=text2IDs("Every effort moves you", T), 
                      n_tokens_target=30, 
                      window_size=GPT_CONFIG_124M["context_length"])
#print(IDs2text(toks, T))

""" Unerstanding Data Genereation"""
vocab = {
"closer": 0,
"every": 1,
"effort": 2,
"forward": 3,
"inches": 4,
"moves": 5,
"pizza": 6,
"toward": 7,
"you": 8,
}    
inverse_vocab = {v:k for k,v in vocab.items()}
# Suppose is the outpos of the model
next_toks_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
probas = torch.softmax(next_toks_logits, dim=0)
next_token = torch.argmax(probas).item()
# print(inverse_vocab[next_token])

# %%
""" Replace ARGMAX with MULTINOMIAL """
torch.manual_seed(123)

next_token = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token])

def print_sampled_toks(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1000)]
    ids_map_freq = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(ids_map_freq):
        print(f"{freq} x {inverse_vocab[i]}")
print_sampled_toks(probas)

#%%
""" Illustration softmax with temepratur scaling """
temperatures = [1, 0.1, 5]
scaled_probas = [torch.softmax(next_toks_logits/t, dim=-1) for t in temperatures]    

import matplotlib.pyplot as plt
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5,3))
for i, t in enumerate(temperatures):
    rects = ax.bar(x+1*bar_width, scaled_probas[i], bar_width, label=f"Temperature = {t}")
ax.set_ylabel("Probability")
ax.set_xticks(x)
ax.set_xlabel(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

# %%
""" Illustration top-k for sampling """
top_k = 3
top_logits, top_pos = torch.topk(next_toks_logits, top_k)
mask_logits = torch.where(
                condition=next_toks_logits < top_logits[-1],
                input=torch.tensor(float("-inf")),
                other=next_toks_logits
                )
topk_probas = torch.softmax(mask_logits, dim=-1)

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

if __name__ == "__main__":
        
    toks_ = generate(M=M, 
                    idx=text2IDs("Every effort moves you ", T),
                    n_generate=30,
                    context_size=GPT_CONFIG_124M["context_length"],
                    top_k=25,
                    temperature=4
                    )
    text = IDs2text(toks_, T)
    print(f"text generated: {text}")
                                                      
# %%
# torch.save(M.state_dict(), "model.pth")
torch.save({
     "model_state_dict": M.state_dict(),
     "optimizer_state_dict": OPT.state_dict()
     },
    "model_and_optimizer.pth"
    )


   