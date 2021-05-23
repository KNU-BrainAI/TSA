import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class EEGNet(nn.Module):
    
    def __init__(self):
        super(EEGNet, self).__init__()
        # Conv2D Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64))
        self.batchnorm1 = nn.BatchNorm2d(8, False)
        # Depthwise Layer
        self.depthwise = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(60, 1),
                                   groups=8)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling1 = nn.AvgPool2d(1, 4)
        # Separable Layer
        self.separable1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,16),
                                    groups=16)
        self.separable2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1))
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.AvgPool2d(1, 8)

        #Flatten
        self.flatten = nn.Flatten()
        
        #Linear
        self.linear1 = nn.Linear(80,4)
    

    def forward(self, x):

        print("input data", x.size())
        # Conv2D
        x = F.pad(x,(31,32,0,0))
        x = self.conv1(x)
        print("conv1", x.size())
        x = self.batchnorm1(x)    
        print("batchnorm", x.size())

        # Depthwise conv2D
        x = self.depthwise(x)
        print("depthwise", x.size())
        x = F.elu(self.batchnorm2(x))
        print("batchnorm & elu", x.size())
        x = self.pooling1(x)
        print("pooling", x.size())
        x = F.dropout(x, 0.5)
        print("dropout", x.size())
        
        # Separable conv2D
        x = F.pad(x,(7,8,0,0))
        x = self.separable1(x)
        x = self.separable2(x)
        print("separable", x.size())
        x = F.elu(self.batchnorm3(x))
        print("batchnorm & elu", x.size())
        x = self.pooling2(x)
        print("pooling", x.size())
        x = F.dropout(x, 0.5)
        print("dropout", x.size())
        
        #Flatten
        x = self.flatten(x)
        print("flatten", x.size())
        
        #Linear
        x = self.linear1(x)
        print("linear", x.size())


        # softmax
        x = F.softmax(x, dim=1)
        x = x.max(dim=1)[1]
        print("softmax : ", x )


        return x
model = EEGNet()


