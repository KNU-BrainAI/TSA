import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# mne imports
import mne
from mne import io
from mne.datasets import sample

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# EEGNet-specific imports
from tensorflow.keras.callbacks import ModelCheckpoint
import models 

data_path = sample.data_path()
data = []

# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)
labels = epochs.events[:, -1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()*1000  # format is in (trials, channels, samples)
y = labels

kernels, chans, samples = 1, 60, 151

# take 50/25/25 percent of the data to train/validate/test
X_train = X[0:144, ]
Y_train = y[0:144]
X_validate = X[144:216, ]
Y_validate = y[144:216]
X_test = X[216:, ]
Y_test = y[216:]

# Numpy array to Tensor
X_train = torch.Tensor(X_train)
Y_train = torch.Tensor(Y_train)
Y_train = F.one_hot(Y_train.to(torch.int64)-1, 4)

X_validate = torch.Tensor(X_validate)
Y_validate = torch.Tensor(Y_validate)
Y_validate = F.one_hot(Y_validate.to(torch.int64)-1, 4)

X_test = torch.Tensor(X_test)
Y_test = torch.Tensor(Y_test)
print("xtrian shape:",X_train.shape)
X_train = X_train.reshape(X_train.shape[0], kernels, chans, samples)
print("xtrian shape:",X_train.shape)
X_validate = X_validate.reshape(X_validate.shape[0], kernels, chans, samples)
X_test = X_test.reshape(X_test.shape[0], kernels, chans, samples)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

trn = data_utils.TensorDataset(X_train, Y_train)
trn_loader = data_utils.DataLoader(trn, batch_size=16, shuffle=True)

val = data_utils.TensorDataset(X_validate, Y_validate)
val_loader = data_utils.DataLoader(val, batch_size=16, shuffle=True)


#################### model training ####################
criterion = nn.CrossEntropyLoss
learning_rate = 0.001
model = models.Test()
model = model.to('cuda')
print(model.parameters)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 100
trn_loss = []
val_loss = []
trn_acc = []
val_acc = []



#%% 
avg_loss = 0
total_batch = len(trn_loader)
loss_test = []

for epoch in range(num_epochs):  # epoch
        model.train()
        avg_loss = 0
        avg_loss2 = 0
        acc = 0
        acc2 = 0
        for data_x, data_y in trn_loader: # iteration
            x,y = data_x.to('cuda'), data_y.to('cuda')
            pred = F.softmax(model(x), dim=1)
            #accuracy
            prediction = torch.max(pred,1)[1]
            y = torch.max(y,1)[1]
            acc += (prediction == y).sum()
            accuracy = acc / len(X_train)
            loss = criterion()(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss / len(trn_loader)
           
        state = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        #torch.save(model.state_dict(), savepath)
        #model.load_state_dict(torch.load('C:/Users/PC/PycharmProjects/TSA/testmodel.pth'))
    
        model.eval()
        with torch.no_grad():
            for data_x, data_y in val_loader:
                val_x, val_y = data_x.to('cuda'), data_y.to('cuda')
                pred2 = F.softmax(model(val_x), dim=1)
                prediction2 = torch.max(pred2,1)[1]
                val_y = torch.max(val_y,1)[1]
                loss2 = criterion()(pred2, val_y)
                avg_loss2 += loss2 / len(val_loader)
                acc2 += (prediction2 == val_y).sum()
                accuracy2 = acc2 / len(X_validate)
    
        print(
            '[Epoch:{}] trn_loss={:.5f}, trn_acc={:.5f}, val_loss={:.5f}, val_acc{:.5f}'.format(epoch + 1, avg_loss,
                                                                                                accuracy, avg_loss2,
                                                                                                accuracy2))
        trn_loss.append(avg_loss.item())
        trn_acc.append(accuracy.item())
        val_loss.append(avg_loss2.item())
        val_acc.append(accuracy2.item())
        #early stopping
          



plt.plot(trn_acc, 'r')
plt.plot(trn_loss,'r,--')
plt.plot(val_acc, 'g')
plt.plot(val_loss, 'g,--')
plt.xlabel('epoch')
plt.title('Training & Validation')
plt.legend(['trn_acc','trn_loss','val_acc','val_loss'])
plt.show()
model.eval()
with torch.no_grad():
    Xtest = X_test.to('cuda')
    Ytest = Y_test.to('cuda')
    prob = F.softmax(model(Xtest), dim=1)
    probs = torch.max(prob,1)[1] + 1
    print(probs)
    print(Ytest)
    test_acc = 0
    test_acc += (probs == Ytest).sum()
    test_acc2 = test_acc / len(Y_test)
    print("---------------------------------------")
    print("test accuracy={}".format(test_acc2))