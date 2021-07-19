import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class EEGNet(nn.Module):
    
    def __init__(self):
        super(EEGNet, self).__init__()
        # Conv2D Layer
#kernel length = sampling rate / 2

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 256)),
            nn.BatchNorm2d(16, False)
            )
 
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(64, 1),groups=16),
            nn.BatchNorm2d(32, False),
            nn.AvgPool2d(1, 16)
            )
 
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,32), groups=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1)),
            nn.BatchNorm2d(32, False),
            nn.AvgPool2d(1, 32)
            )
 
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(32*4,2)


    def forward(self, x):

        # Conv2D
        x = F.pad(x,(127,128,0,0))
        x = self.layer1(x)

        # Depthwise conv2D
        x = F.elu(self.layer2(x))
        x = F.dropout(x, 0.5)
        
        # Separable conv2D
        x = F.pad(x,(15,16,0,0))
        x = F.elu(self.layer3(x))
        x = F.dropout(x, 0.5)
        
        #Flatten
        x = self.flatten(x)
        
        #Linear
        x = self.linear1(x)

        return x
    
    

class ConstrainedLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight.clamp(min=-0.25, max=0.25), self.bias)

class Deep_ConvNet(nn.Module):
    def __init__(self, bias=False, num_class=2):
        super(Deep_ConvNet, self).__init__()

        self.conv_split = nn.Sequential(
            nn.Conv2d(1, 25, (1,10), 1),
            nn.Conv2d(25, 25, (64,1), 1, bias=False),
            )
        self.post_conv = nn.Sequential(
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_pool1 = nn.Sequential(
            nn.Conv2d(25, 50, (1,10), 1, bias=False),
            nn.BatchNorm2d(50),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_pool2 = nn.Sequential(
            nn.Conv2d(50, 100, (1,10), 1, bias=False),
            nn.BatchNorm2d(100),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_pool3 = nn.Sequential(
            nn.Conv2d(100, 200, (1,10), 1, bias=False),
            nn.BatchNorm2d(200),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_fc = nn.Sequential(
            ConstrainedLinear(200*14*1, num_class)
            )

    def forward(self, x):
        out = self.conv_split(x)
        out = self.post_conv(out)
        out = self.conv_pool1(out)
        out = self.conv_pool2(out)
        out = self.conv_pool3(out)
        out = out.view(-1, np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        return out