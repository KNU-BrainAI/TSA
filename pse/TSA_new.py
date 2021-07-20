import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import os, fnmatch
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# mne imports
import mne

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# EEGNet-specific imports
import EEGModels_torch

model = EEGModels_torch.Deep_ConvNet()
model = model.to('cuda')

data_path = "C:/Users/PC/Desktop/SSSEP/new_data"
files = fnmatch.filter(os.listdir(data_path),'*.set')
os.chdir(data_path)
tmin, tmax= 0, 3

X=[]
Y=[]

for k in range(0,32):
    
    raw = mne.io.read_raw_eeglab(files[k])
    
    events, event_id = mne.events_from_annotations(raw)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                            picks=picks, baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1]
    kernels, chans, samples = 1, 64, 1537
        
    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
    data = epochs.get_data()*1000*1000 # format is in (trials, channels, samples)
    y = labels
    
    X.extend(data)
    Y.extend(labels)
        
X=np.array(X)
Y=np.array(Y)
    
print('X_size : ', len(X))
print('Y_size : ', len(Y))
    

# take 50/25/25 percent of the data to train/validate/test
X_train = X[0:800, ]
Y_train = Y[0:800]
X_validate = X[800:1200, ]
Y_validate = Y[800:1200]
X_test = X[1200:, ]
Y_test = Y[1200:]

# Numpy array to Tensor
X_train = torch.Tensor(X_train)
Y_train = torch.Tensor(Y_train)
Y_train = F.one_hot(Y_train.to(torch.int64)-1, 4)

X_validate = torch.Tensor(X_validate)
Y_validate = torch.Tensor(Y_validate)
Y_validate = F.one_hot(Y_validate.to(torch.int64)-1, 4)

X_test = torch.Tensor(X_test)
Y_test = torch.Tensor(Y_test)
# Y_test = F.one_hot(Y_test.to(torch.int64)-1, 4)
print("xtrian shape:",X_train.shape)

X_train = X_train.reshape(X_train.shape[0], kernels, chans, samples)
X_validate = X_validate.reshape(X_validate.shape[0], kernels, chans, samples)
X_test = X_test.reshape(X_test.shape[0], kernels, chans, samples)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

trn = data_utils.TensorDataset(X_train, Y_train)
trn_loader = data_utils.DataLoader(trn, batch_size=64, shuffle=True)

val = data_utils.TensorDataset(X_validate, Y_validate)
val_loader = data_utils.DataLoader(val, batch_size=32, shuffle=False)

# test = data_utils.TensorDataset(X_test, Y_test)
# test_loader = data_utils.DataLoader(test, batch_size=16, shuffle=True)


#################### model training ####################
criterion = nn.CrossEntropyLoss
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 100
trn_loss = []
val_loss = []
trn_acc = []
val_acc = []

savepath = 'C:/Users/PC/Desktop/SSSEP/savepath/testmodel.pth'

for epoch in range(num_epochs): # epoch
    model.train()
    avg_loss = 0
    avg_loss2 = 0
    acc = 0
    acc2 = 0
    
    for data_x, data_y in trn_loader: # iteration
        x,y = data_x.to('cuda'), data_y.to('cuda')
        optimizer.zero_grad()
        pred = F.softmax(model(x), dim=1)
        #accuracy
        prediction = torch.max(pred,1)[1]
        y = torch.max(y,1)[1]
        acc += (prediction == y).sum()
        accuracy = acc / len(X_train)
        #print(F.softmax(pred))
        loss = criterion()(pred, y)
        #print("loss: ",loss)
        loss.backward()
        optimizer.step()
        avg_loss += loss / len(trn_loader)
        
    state={"state_dict" : model.state_dict(), "optimizer" :  optimizer.state_dict()}
    torch.save(model.state_dict(),savepath)
    model.load_state_dict(torch.load('C:/Users/PC/Desktop/SSSEP/savepath/testmodel.pth'))
    
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
            
    print('[Epoch:{}] trn_loss={:.5f}, trn_acc={:.5f}, val_loss={:.5f}, val_acc{:.5f}'.format(epoch+1,avg_loss,accuracy,avg_loss2,accuracy2))    
    trn_loss.append(avg_loss.item())
    trn_acc.append(accuracy.item())
    val_loss.append(avg_loss2.item())
    val_acc.append(accuracy2.item())
print("finish training & Validation!")


plt.plot(trn_acc, 'r')
plt.plot(trn_loss,'r,--')
plt.plot(val_acc, 'b')
plt.plot(val_loss, 'b,--')
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
  test_acc = 0
  test_acc += (probs == Ytest).sum()
  test_acc2 = test_acc / len(Y_test)
  print("---------------------------------------")
  print("test accuracy={}".format(test_acc2))
  
