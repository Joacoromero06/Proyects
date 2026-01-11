#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 11:05:52 2026

@author: joaco
"""
import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
        
            torch.nn.Linear(n_input, 30), 
            torch.nn.ReLU(),
            
            torch.nn.Linear(30, 20), 
            torch.nn.ReLU(),
            
            torch.nn.Linear(20, n_output)            
            )
    def forward(self, X):
        logits = self.layers(X)
        return logits
# %%

import torch
from torch.utils.data import Dataset
class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.features = x
        self.labels = y
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    def __len__(self):
        return self.features.shape[0]
# %%
import torch
from torch.utils.data import DataLoader

x_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
        ])
y_train = torch.tensor([0,0,0,1,1])
x_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
        ])
y_test = torch.tensor([0,1])
    
train_ds = ToyDataset(x_train, y_train)
test_ds = ToyDataset(x_test, y_test)

torch.manual_seed(123)
train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0          
    ) 
# %%
import torch.nn.functional as F
torch.manual_seed(123)
model = NeuralNetwork(2, 2)
optimizer = torch.optim.SGD(
        model.parameters(), lr=0.5
    )
n_epochs = 3
for e in range(n_epochs):
    model.train()
    
    for batch_i, (feature, label) in enumerate(train_loader):
        logits = model(feature)
        loss = F.cross_entropy(logits, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch: {e+1:03d}/{n_epochs:03d} '
              f'| Batch: {batch_i:03d}/{len(train_loader):03d} '
              f'| Train loss: {loss:.2f}'
              )

    model.eval()    
    
model.eval()
with torch.no_grad():
    outputs = model(x_train)
    
torch.set_printoptions(sci_mode=True)
probas = torch.softmax(outputs, dim=1) 
predictions = torch.argmax(probas, dim=1)

# %%
def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0
    total_eg = 0
    
    for i, (f, l) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(f)
        predictions = torch.argmax(logits, dim=1)
        
        compare = (l == predictions)
        correct += torch.sum(compare)
        total_eg += len(compare)
    
    return (correct/total_eg).item()

