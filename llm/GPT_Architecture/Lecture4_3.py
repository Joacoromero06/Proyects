#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 17:00:03 2026

@author: joaco
"""
import torch
from torch import nn
class ExampleDNN(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [nn.Sequential( nn.Linear(layer_sizes[0], layer_sizes[1]), nn.GELU() ),
             nn.Sequential( nn.Linear(layer_sizes[1], layer_sizes[2]), nn.GELU() ),
             nn.Sequential( nn.Linear(layer_sizes[2], layer_sizes[3]), nn.GELU() ),
             nn.Sequential( nn.Linear(layer_sizes[3], layer_sizes[4]), nn.GELU() ),
             nn.Sequential( nn.Linear(layer_sizes[4], layer_sizes[5]), nn.GELU() )]
            )
    def forward(self, X):
        for L in self.layers:
            Y = L(X)
            if self.shortcut and X.shape == Y.shape :
                X = X + Y
            else:
                X = Y
        return X
torch.manual_seed(123)
layer_sizes = [3,3,3,3,3,1]
x = torch.tensor([1., 0., -1.])
model = ExampleDNN(layer_sizes, use_shortcut=True)

def print_gradients(model, X):
    Y = model(X)
    Y_true = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(Y, Y_true)
    loss.backward()
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'{name} has gradient mean of {param.grad.abs().mean().item()}')
print_gradients(model, x)

    
    
    
    
    
    



