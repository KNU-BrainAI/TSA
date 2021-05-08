
import mne
import os, fnmatch
import torch

dPath = "D:\Dropbox\SSSEP\Data"
files=fnmatch.filter(os.listdir(dPath),'*.set')

raw = mne.io.read_raw_eeglab(files[0])
#raw.plot()

# add plot
raw.plot()

events, event_id = mne.events_from_annotations(raw)



