#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 08:49:27 2026

@author: joaco
"""

import torch
import torch.nn.functional as F
#from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = w1 * x1 + b
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

loss.backward()
print(w1.grad)
print(b.grad)


