import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class EEGNet(nn.Module):
    
    def __init__(self):
        super(EEGNet, self).__init__()
    
        # Conv2D Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64))
        self.batchnorm1 = nn.BatchNorm2d(8, False)
        
        # Depthwise Layer
        self.depthwise = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(8, 1),
                                   groups=8)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling1 = nn.AvgPool2d(1, 4)
        
        # Separable Layer
        self.separable = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16))
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.AvgPool2d(1, 8)

        #Flatten
        self.flatten = nn.Flatten()
        
        #linear
        self.linear1 = nn.Linear(96,4)
        

    def forward(self, x):

        print("연산 전", x.size())
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
        x = self.separable(x)
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
        
        # FC Layer
        x = F.softmax(self.linear1(x))
        print("linear", x.size())
        
        return x
    
model = EEGNet()
myModel = model(torch.randn(10,1,64,128))

"""    
from torchsummary import summary
model = EEGNet()
summary(model, input_size=(1,64,128), batch_size = 1) #summary(your_model, input_size=(channels, H, W))
"""

#%% same padding
""" 
in_height, in_width = 64,128
filter_height, filter_width = 8,1
strides=(None,1,1)
out_height = np.ceil(float(in_height) / float(strides[1]))
out_width  = np.ceil(float(in_width) / float(strides[2]))

print(out_height)
print(out_width)

#The total padding applied along the height and width is computed as:

if (in_height % strides[1] == 0):
  pad_along_height = max(filter_height - strides[1], 0)
else:
  pad_along_height = max(filter_height - (in_height % strides[1]), 0)
if (in_width % strides[2] == 0):
  pad_along_width = max(filter_width - strides[2], 0)
else:
  pad_along_width = max(filter_width - (in_width % strides[2]), 0)

print(pad_along_height, pad_along_width)
  
#Finally, the padding on the top, bottom, left and right are:

pad_top = pad_along_height // 2
pad_bottom = pad_along_height - pad_top
pad_left = pad_along_width // 2
pad_right = pad_along_width - pad_left

print(pad_left, pad_right, pad_top, pad_bottom)
"""

#%% Summary
"""
input torch.Size([2, 1, 64, 128])
conv2D torch.Size([2, 8, 64, 65])
batchnorm torch.Size([2, 8, 64, 65])
depthwise torch.Size([2, 16, 57, 65])
batchnorm torch.Size([2, 16, 57, 65])
pooling1 torch.Size([2, 16, 15, 17])
dropout torch.Size([2, 16, 15, 17])
pointwise torch.Size([2, 16, 15, 2])
batchnorm torch.Size([2, 16, 15, 2])
pooling2 torch.Size([2, 16, 2, 1])
dropout torch.Size([2, 16, 2, 1])
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 64, 65]             520
       BatchNorm2d-2            [-1, 8, 64, 65]              16
            Conv2d-3           [-1, 16, 57, 65]             144
       BatchNorm2d-4           [-1, 16, 57, 65]              32
         AvgPool2d-5           [-1, 16, 15, 17]               0
            Conv2d-6            [-1, 16, 15, 2]           4,112
       BatchNorm2d-7            [-1, 16, 15, 2]              32
         AvgPool2d-8             [-1, 16, 2, 1]               0
================================================================
Total params: 4,856
Trainable params: 4,856
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 1.45
Params size (MB): 0.02
Estimated Total Size (MB): 1.50
----------------------------------------------------------------
"""
