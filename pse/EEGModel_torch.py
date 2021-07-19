import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class EEGNet(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 16, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(EEGNet,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d( ( ( (kernLength-1 //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        # Depthwise Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1*D, kernel_size=(Chans, 1), groups=F1),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(1, 16)
        )
        # Separable Layer
        self.layer3 = nn.Sequential(
            #Padding size = 1st Conv2D Filter(W,H) -> Pad((H-1)/2,(H-1)/2,(W-1)/2,(W-1)/2)
            nn.ZeroPad2d(((((32 - 1) // 2 ) + 1), ( (32 - 1) // 2), 0, 0)),
            nn.Conv2d(in_channels=F1*D, out_channels=F2, kernel_size=(1, 32), groups=F1*D),
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, 1)),
            nn.BatchNorm2d(F2, False),
            nn.AvgPool2d(1, 32)
        )
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(F2 * 4 , class_num)

    def forward(self, x):
        # Conv2D
        y = self.layer1(x)
        # Depthwise conv2D
        y = F.elu(self.layer2(y))
        y = F.dropout(y, 0.5)
        # Separable conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y
model = EEGNet()