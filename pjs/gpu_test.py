# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:25:17 2021

@author: PC
"""
import torch
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)