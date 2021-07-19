import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class EEGNet(nn.Module):
    
    def __init__(self):
        super(EEGNet, self).__init__()
        # Conv2D Layer

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64)),
            nn.BatchNorm2d(8, False)
            )
 
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(60, 1),groups=8),
            nn.BatchNorm2d(16, False),
            nn.AvgPool2d(1, 4)
            )
 
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,16), groups=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1)),
            nn.BatchNorm2d(16, False),
            nn.AvgPool2d(1, 8)
            )
 
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(16*5,4)


    def forward(self, x):

        # Conv2D
        x = F.pad(x,(31,32,0,0))
        x = self.layer1(x)

        # Depthwise conv2D
        x = F.elu(self.layer2(x))
        x = F.dropout(x, 0.5)
        
        # Separable conv2D
        x = F.pad(x,(7,8,0,0))
        x = F.elu(self.layer3(x))
        x = F.dropout(x, 0.5)
        
        #Flatten
        x = self.flatten(x)
        
        #Linear
        x = self.linear1(x)

        return x
model = EEGNet()