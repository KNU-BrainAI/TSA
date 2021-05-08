import numpy as np
import os, fnmatch

# mne imports
import mne
from mne import io

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

#default of data
data=[]
k=0


# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

############ Process, filter and epoch the data ################
# Set parameters and read data
data_path = "C:/Users/oo/Desktop/SSSEP/Research/Data/SSSEP"
files = fnmatch.filter(os.listdir(data_path),'*.set')
os.chdir(data_path)
tmin, tmax= 0, 3

while k < 16: 
    raw = mne.io.read_raw_eeglab(files[k])
    events, event_id = mne.events_from_annotations(raw)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, 
                        picks=picks, baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1]
        
    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
    X = epochs.get_data()*1000*1000 # format is in (trials, channels, samples)
    y = labels
    
    kernels, chans, samples = 1, 64, 1537
        
    # take 50/20/30 percent of the data to train/validate/test
    X_train      = X[25:75,]
    Y_train      = y[25:75]
    X_validate   = np.concatenate([X[15:25], X[75:85]])
    Y_validate   = np.concatenate([y[15:25], y[75:85]])
    X_test       = np.concatenate([X[0:15], X[85:100]])
    Y_test       = np.concatenate([y[0:15], y[85:100]])
        
    ############################# EEGNet portion ##################################
    
    # convert labels to one-hot encodings.
    Y_train      = np_utils.to_categorical(Y_train-1)
    Y_validate   = np_utils.to_categorical(Y_validate-1)
    Y_test       = np_utils.to_categorical(Y_test-1)
    
    # convert data to NHWC (trials, channels, samples, kernels) format. Data 
    # contains 64 channels and 1537 time-points. Set the number of kernels to 1.
    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
            
    #%%    
    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
                   dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                   dropoutType = 'Dropout')
        #%%
       # compile the model and set the optimizers.
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                      metrics = ['accuracy'])
        
        # count number of parameters in the model
    numParams    = model.count_params()    
        
       # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='C:/Users/oo/Desktop/SSSEP/tmp/checkpoint.h5',
                                   verbose=1, save_best_only=True)
        
        ##########################################################################
        # if the classification task was imbalanced (significantly more trials in one
        # class versus the others) you can assign a weight to each class during 
        # optimization to balance it out. This data is approximately balanced so we 
        # don't need to do this, but is shown here for illustration/completeness. 
        ##########################################################################
        # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
        # the weights all to be 1
    class_weights = {0:1, 1:1}
        
        ###########################################################################
        #fit the model.
        ###########################################################################
    fittedModel = model.fit(X_train, Y_train, batch_size = 8, epochs = 20, 
                            verbose = 2, validation_data=(X_validate, Y_validate),
                            callbacks=[checkpointer], class_weight = class_weights)
    
        # load optimal weights
    model.load_weights(r'C:/Users/oo/Desktop/SSSEP/tmp/checkpoint.h5')
        
        ###############################################################################
        # can alternatively used the weights provided in the repo. If so it should get
        # you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
        # system.
        ###############################################################################
    
        # WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
        # model.load_weights(WEIGHTS_PATH)
    
        ###############################################################################
        # make prediction on test set.
        ###############################################################################
    
    probs       = model.predict(X_test)
    preds       = probs.argmax(axis = -1)
    acc         = np.mean(preds == Y_test.argmax(axis=-1))
    print("\n********%d번째 피험자********\nClassification accuracy: %f" % (k+1,acc))
        
    # plot the accuracy and loss graph
    plt.figure(0)
    plt.subplot(2,1,1)
    plt.plot(fittedModel.history['accuracy'],':')
    plt.plot(fittedModel.history['val_accuracy'])
    plt.title('acc')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='upper left')
    
    plt.subplot(2,1,2)
    plt.plot(fittedModel.history['loss'],':')
    plt.plot(fittedModel.history['val_loss'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.legend(['loss','val_loss'], loc='upper left')
    plt.show()
#%%
    plt.figure(0)
    plt.plot(fittedModel.history['accuracy'],'r,-.')
    plt.plot(fittedModel.history['val_accuracy'],'r')
    plt.plot(fittedModel.history['loss'],'b,-.')
    plt.plot(fittedModel.history['val_loss'],'b')
    plt.title('acc & loss')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc','loss','val_loss'], loc='upper left')
    plt.show()
#%% 
     
    names        = ['left', 'right']
    plt.figure(1)
    plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
    plt.show()
    
    #list of accuracy data
    data.append(acc)
    k = k+1
    
#total data
a=round(np.mean(data),5)
b=round(np.std(data),5)
print("Mean of the accuracy : %f \nStandard deviation of the accuracy : %f" %(a,b))

plt.bar(np.arange(16), data)
plt.xticks(np.arange(16),np.arange(1,17))
plt.title('all of the subjects')
plt.xlabel('subject')
plt.ylabel('accuracy')


#learning rate 줄여보기
