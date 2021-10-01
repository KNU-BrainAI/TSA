# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:47:50 2021

@author: PC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import early_stopping
from early_stopping import EarlyStopping
import numpy as np
import os
import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# mne imports
import mne
from mne import io
from mne.datasets import sample
import math
# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# EEGNet-specific imports
import models
from sklearn.model_selection import train_test_split, KFold

import numpy as np
from sklearn.model_selection import LeavePGroupsOut

groups = []
for x in range(1,11):
    for y in range(1,11):
        groups.insert(-1,y)
  
lpgo = LeavePGroupsOut(n_groups=2)

# GPU allocation
save_dir = 'C:/Users/PC/Desktop/TSA_result/EEGNet_Within/3_EEGNet_10C3CV/'
result_txt = '3_EEGNet_10C3CV.txt'

##################### Process Main  ######################
for i in range(1,2):
    X = []
    Y =[]
    for filename in glob.glob('C:/Users/PC/Desktop/matlab/dataset/TSA_Raw/seq_sub'+str(i)+'000/*_fr.set'):
        #data path where preprocessed data
        dpath = filename
        print(dpath)
        #get data and find event
        eeglab_raw  = mne.io.read_raw_eeglab(dpath)
        print(eeglab_raw.annotations[1])
        print(len(eeglab_raw.annotations))
        print(set(eeglab_raw.annotations.duration))
        print(set(eeglab_raw.annotations.description))
        print(eeglab_raw.annotations.onset[1])
        anno = mne.read_annotations(dpath)
        print(anno)
        (events_from_annot,event_dict) = mne.events_from_annotations(eeglab_raw)
        print(event_dict)
        print(events_from_annot[:])
        event_id = dict(left=1,right=2)
        tmin = 0
        tmax = 2.9980
        epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id, tmin, tmax,baseline = None)
        labels = epochs.events[:,-1]
        #data *1000 uV to V
        data = epochs.get_data( )*1000000 # format is in (trials, channels, samples)
        X.extend(data)
        Y.extend(labels)
    X=np.array(X)
    Y=np.array(Y)
    print(len(X))
    print(len(Y))
    #samples = 3sec * 512Hz sampling rate
    kernels, chans, samples = 1, 64, 1536
    

    for fold, (train_index, test_index) in enumerate(lpgo.split(X, Y, groups=groups)):
        print("fold:",fold,"TRAIN:",len( train_index), "TEST:", len(test_index)) 
        
