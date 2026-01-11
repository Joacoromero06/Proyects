#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:57:45 2026

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
    def foreward(self, X):
        logits = self.layers(X)
        return logits
model = NeuralNetwork(50, 3)
X = torch.rand((1,50))
out = model.foreward(X)