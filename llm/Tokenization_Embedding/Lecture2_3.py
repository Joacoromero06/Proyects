#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 06:42:43 2026

@author: joaco
"""

import os
abs_path = os.path.abspath(__file__)
cname = os.path.dirname(abs_path)
os.chdir(cname)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# %%
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        """
        context size is called: max_legth
        step inside the text is called: stride
        """
        self.inputs_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)
        
        for i in range(0, len(token_ids)-max_length, stride):
            # input-ith correspond to target-ith
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            
            # Convert Python list to Torch.Tensor 
            self.inputs_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.inputs_ids)
    def __getitem__(self, i):
        return self.inputs_ids[i], self.target_ids[i]    

# %%
import tiktoken
def create_dataloader_v1(text, 
                         batch_size=4, 
                         max_length=256,
                         stride=128,
                         shuffle=True,
                         drop_last=True,
                         num_workers=0):
    """
    text: data which be extracted for GPTDataset
    batch_size: n° of (input, target) per batch
    max_legth: length of input and target tensors, defines context size
    stride: n° position, input shift across batches
    """
    # Create BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create Dataset for purpose
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    
    # Create DataLoader using hyperparameters
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader
  
                  
# %%
dataloader = create_dataloader_v1(raw_text,
                                  batch_size=8,
                                  max_length=4,
                                  stride=4,
                                  shuffle=False,
                                  )
iter_data = iter(dataloader)
print(next(iter_data))
print(next(iter_data))

