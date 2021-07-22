# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:18:28 2021

@author: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import os
import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# mne imports
import mne
from mne import io
from mne.datasets import sample

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# EEGNet-specific imports
import models
from sklearn.model_selection import train_test_split, KFold

model = models.EEGNet()
model = model.to('cuda')

# GPU allocation


kf = KFold(n_splits=5, shuffle= True, random_state= True)
##################### Process Main  ######################

X = [1,2,3,4,5,6,7,8,9,10]
Y = [11,12,13,14,15,16,17,18,19,20]
X=np.array(X)
Y=np.array(Y)
for fold, (train_index, test_index) in enumerate(kf.split(X)):
     X_train = X[train_index]
     Y_train = Y[train_index]
    
     X_test = X[test_index]
     Y_test = Y[test_index]
     
     X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, train_size=0.7, shuffle=True,random_state=1004)
     print("k=",fold)
     print("Train")      
     print(X_train)                                       
     print(Y_train)
     print("VAL")
     print(X_validate)
     print(Y_validate)
     print("test")
     print(X_test)
     print(Y_test)