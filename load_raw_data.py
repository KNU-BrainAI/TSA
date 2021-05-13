
from BCI2kReader import BCI2kReader as b2k
import matplotlib.pyplot as plt
import numpy as np
import os
import mne

DATA_PATH = 'D:\OneDrive - knu.ac.kr\BrainAI\Research\Data\TSA'
CODE_PATH = 'D:\OneDrive - knu.ac.kr\Python\TSA'
SubList = os.listdir(DATA_PATH)
FileList = os.listdir(DATA_PATH + '\\' + SubList[0])
FileName = DATA_PATH + '\\' + SubList[0] + '\\' + FileList[1]

with b2k.BCI2kReader(FileName) as data: #opens a stream to the dat file
    srate=data.samplingrate
    raw=data.signals
    state=data.states

print(raw.shape) # EEG: 1-64, EMG: 65-68, and time
print(state.keys()) # print keys only in dict

print(state.get('StimulusCode'))
StimCode = state.get('StimulusCode') 
print(np.unique(StimCode)) # left : 1 , right : 2
idx=np.where(StimCode==1)

StimType=state.get('StimulusTime')
print(np.unique(StimType)) 
idx=np.where(StimType==1)



