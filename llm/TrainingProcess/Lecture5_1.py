#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:37:05 2026

@author: joaco
"""


import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(cur_dir, '..'))
print("root",  proj_root)
sys.path.append(proj_root)

import torch, tiktoken
torch.manual_seed(123)
from GPT_Architecture.Lecture4_5 import GPTModel, GPT_CONFIG_124M, generate_text1

T = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)

#batch = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (2,4))
#out = model(batch)

# %%

def text2IDs(text, T):
    return torch.tensor(T.encode(text, allowed_special={'<|EOF|>'})).unsqueeze(0)
def IDs2text(ids, T):
    return T.decode(ids.squeeze(0).tolist())

ids = generate_text1(model, 
                     idx            =text2IDs("Every step moves ", T) , 
                     n_tokens_target=10, 
                     window_size    =GPT_CONFIG_124M["context_length"])
generated = IDs2text(ids, T)

# %%

inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                       [40,1107, 588]       # "I really like"]
                       ])
targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
                       [1107, 588, 11311]])# " really like chocolate"])
with torch.no_grad():
    Y = model(inputs)
probas = torch.softmax(Y, dim=-1)

next_ids = torch.argmax(probas, dim=-1, keepdim=True)
print('target batch: ', IDs2text(targets[0], T) ) 
print('output batch: ', IDs2text(next_ids[0].flatten(), T) ) 

text_idx = 0
target_probas0 = probas[text_idx, [0,1,2], targets[text_idx]]

text_idx = 1
target_probas1 = probas[text_idx, [0,1,2], targets[text_idx]]

# logits (b,n,vocab). Output model
# probas (b,n,vocab). Probability distribution of entire distribution
# target_prob_i (n), i=1..b. Probability of target tokens
"""
We would want target_prob_i [1e-5, ..., 2.1e-4] be higher possible and 
happens at same time, that is prod[i=1..n] P(Xi) = 1. 
Because vocab size is big, P(Xi) -> 0, Then the product comes to zero fast.
Applying logarithm, products become sums, So the operation to know the 
probability of the model given his distribution output, be the target is 
obtained via mean of the values in target_prob_i applying logarithm
We know small values becomes negative via logarithm, so we use the opposite
of the mean, The mean represent the noise of the answer, the prob of the 
model be 100% correct.
Because the perfecttion is when prod[i=1..n] P(Xi) = 1, with logarithm this 
is, mean(log(P(Xi), i=1..n)) = 0. So we want to minimize the loss calculated
as the mean of the logarithm of the probabilities of the target_tokens in
the model given distribution
"""

# log_prob (n*b). log() Probability of target tokens of each batch
log_probas = torch.log( torch.cat((target_probas0, target_probas1)) ) 

avarage_log_prob = torch.mean(log_probas)
# Level of confusion 
cross_entropy_loss = -avarage_log_prob
# Number of word the model is confused to choose to predict
perplexity = torch.exp(cross_entropy_loss)


# %% 
# Y(b,n,vocab)
logits_flat = Y.flatten(0,1) # (b*n, vocab)
# targets(b,n)
targets_flat = targets.flatten() # (b*n)
# Compute loss by cross entropy
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)



