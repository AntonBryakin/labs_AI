# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:56:59 2025

@author: rezuqq
"""
import numpy as np
import random
import torch

x=torch.randint(0, 10, (5, 3))
print(x)

x_to_float32 = x.type(torch.float32)
print(x_to_float32)

y = x_to_float32**2 
print(y)

a = random.randint(1, 10)
z = x_to_float32*a
print("рандомное число = ",a)
print(z)

torch.set_printoptions(precision= 4, sci_mode= False)
b = torch.exp(x_to_float32)
print(b)
