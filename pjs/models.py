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
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
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
        self.linear1 = nn.Linear(F2 * 3 , class_num)

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

class EEGNet_1_split(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 16, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(EEGNet_1_split,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        # Depthwise Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1*D, kernel_size=(Chans, 1), groups=F1),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(1, 8)
        )
        # Separable Layer
        self.layer3 = nn.Sequential(
            #Padding size = 1st Conv2D Filter(W,H) -> Pad((H-1)/2,(H-1)/2,(W-1)/2,(W-1)/2)
            nn.ZeroPad2d(((((32 - 1) // 2 ) + 1), ( (32 - 1) // 2), 0, 0)),
            nn.Conv2d(in_channels=F1*D, out_channels=F2, kernel_size=(1, 32), groups=F1*D),
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, 1)),
            nn.BatchNorm2d(F2, False),
            nn.AvgPool2d(1, 16)
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

class EEGNet_1_weight(nn.Module):
    
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    
    def __init__(self, bias = False, class_num = 2, F1 = 16, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(EEGNet_1_weight,self).__init__()
        
        self.weight =  nn.Parameter(torch.tensor(torch.rand((64,1536), requires_grad = True)))
        
        
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
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
        self.linear1 = nn.Linear(F2 * 3 , class_num)
        
       
    def forward(self, x):
        y = x * self.weight
        # Conv2D
        y = self.layer1(y)
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
class EEGNet_1_sum(nn.Module):
    
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    
    def __init__(self, bias = False, class_num = 2, F1 = 16, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(EEGNet_1_sum,self).__init__()
             
        self.skip1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernLength), groups=1,bias=bias),
        )
        
        self.skip2 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength//2)-1) //2) +1),(((kernLength//2)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernLength//2), groups=1,bias=bias),
        )
        
        self.skip3 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength//4)-1) //2) +1),(((kernLength//4)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernLength//4), groups=1,bias=bias),
        )
        
        
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
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
        self.linear1 = nn.Linear(F2 * 3 , class_num)
        
       
    def forward(self, x):
        y1 = self.skip1(x)
        x = x+ y1 
        y2 = self.skip2(x)
        x = x+ y2
        y3 = self.skip3(x)
        x = x + y3
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

class Test(nn.Module):
    
    def __init__(self):
        super(Test, self).__init__()
        # Conv2D Layer
        self.skip1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (64-1) //2) +1),((64-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 64), groups=1),
        )
        
        self.skip2 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((64//2)-1) //2) +1),(((64//2)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 64//2), groups=1),
        )
        
        self.skip3 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((64//4)-1) //2) +1),(((64//4)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 64//4), groups=1),
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64)),
            nn.BatchNorm2d(8, False)
            )
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64))
        # self.batchnorm1 = nn.BatchNorm2d(8, False)
        
        # Depthwise Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(60, 1),groups=8),
            nn.BatchNorm2d(16, False),
            nn.AvgPool2d(1, 4)
            )
        
        # self.depthwise = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(60, 1),
                                   # groups=8)
        # self.batchnorm2 = nn.BatchNorm2d(16, False)
        # self.pooling1 = nn.AvgPool2d(1, 4)
        

        # Separable Layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,16), groups=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1)),
            nn.BatchNorm2d(16, False),
            nn.AvgPool2d(1, 8)
            )
        # self.separable1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,16),
        #                             groups=16)
        # self.separable2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1))
        # self.batchnorm3 = nn.BatchNorm2d(16, False)
        # self.pooling2 = nn.AvgPool2d(1, 8)

        #Flatten
        self.flatten = nn.Flatten()
        
        #Linear

        self.linear1 = nn.Linear(16*5,4)


    def forward(self, x):
        y1 = self.skip1(x)
        x = x+ y1 
        y2 = self.skip2(x)
        x = x+ y2
        y3 = self.skip3(x)
        x = x+ y3
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
    
class Tuto(nn.Module):
    
    def __init__(self):
        super(Tuto, self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64)),
            nn.BatchNorm2d(8, False)
            )
        
        # Depthwise Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(60, 1),groups=8),
            nn.BatchNorm2d(16, False),
            nn.AvgPool2d(1, 4)
            )

        # Separable Layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,16), groups=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1)),
            nn.BatchNorm2d(16, False),
            nn.AvgPool2d(1, 8)
            )


        #Flatten
        self.flatten = nn.Flatten()
        
        #Linear
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
"""
for within-subject : Deep_ConvNEt, EEGNet, EEG-TCNet, CCRNN
for cross-subject : with 'sub_*'
"""


class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return F.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class ConstrainedLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight.clamp(min=-0.25, max=0.25), self.bias)


class Deep_ConvNet(nn.Module):
    def __init__(self, bias=False, num_class=2):
        super(Deep_ConvNet, self).__init__()

        self.conv_split = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), 1),
            nn.Conv2d(25, 25, (32, 1), 1, bias=False),
        )
        self.post_conv = nn.Sequential(
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3), 3),
            nn.Dropout(0.3)
        )
        self.conv_pool1 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 10), 1, bias=False),
            nn.BatchNorm2d(50),
            nn.MaxPool2d((1, 3), 3),
            nn.Dropout(0.3)
        )
        self.conv_pool2 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 10), 1, bias=False),
            nn.BatchNorm2d(100),
            nn.MaxPool2d((1, 3), 3),
            nn.Dropout(0.3)
        )
        self.conv_pool3 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 10), 1, bias=False),
            nn.BatchNorm2d(200),
            nn.MaxPool2d((1, 3), 3),
            nn.Dropout(0.3)
        )
        self.conv_fc = nn.Sequential(
            ConstrainedLinear(200 * 14 * 1, num_class)
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
class EEG_TCNet(nn.Module):
    def __init__(self, bias=False, num_class=2, drop_ratio=.5, F1=8, D=2):
        super(EEG_TCNet, self).__init__()
        F2 = F1*D

        self.conv_temporal = nn.Sequential(
            nn.ZeroPad2d((((250-1)//2)+1, ((250-1)//2), 0, 0)),
            nn.Conv2d(1, F1, (1,250), 1, bias=bias),
            nn.BatchNorm2d(F1),
            )

        self.conv_spatial = nn.Sequential(
            ConstrainedConv2d(F1, F1*D, (32,1), 1, bias=bias, groups=F1),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(drop_ratio)
            )

        self.conv_separable = nn.Sequential(
            nn.ZeroPad2d((((125-1)//2)+1, ((125-1)//2), 0, 0)),
            nn.Conv2d(F1*D, F2, (1,125), 1, bias=bias, groups=F1*D), #depthwise
            nn.Conv2d(F2, F2, 1, 1),  #pointwise = 1dconv
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)), #(12)
            nn.Dropout(drop_ratio)
            )

        self.conv_fc = nn.Sequential(
            ConstrainedLinear(F2*1*15, num_class)
            #nn.Linear(F2*1*15, num_class) #(16*1*10)
            )

        # TCN-block
        self.tcn_block1 = nn.Sequential(
            nn.ZeroPad2d((2,1,0,0)),
            nn.Conv1d(F2, F2, 4, 1),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.ZeroPad2d((2,1,0,0)),
            nn.Conv1d(F2, F2, 4, 1),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),
            )
        self.tcn_block2 = nn.Sequential(
            nn.ZeroPad2d((3,3,0,0)),
            nn.Conv1d(F2, F2, 4, 1, dilation=2),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.ZeroPad2d((3,3,0,0)),
            nn.Conv1d(F2, F2, 4, 1, dilation=2),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),
            )

    def forward(self, x):
        out = self.conv_temporal(x)
        out = self.conv_spatial(out)
        out = self.conv_separable(out)
        out = torch.squeeze(out, axis=2)
        tcn = self.tcn_block1(out)
        out = out + tcn
        out = nn.ELU()(out)
        tcn = self.tcn_block2(out)
        out = out + tcn
        out = nn.ELU()(out)
        out = out.view(-1, np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        return out

class CCRNN(nn.Module):
    def __init__(self, num_classes=2, drop_ratio=0.5, nSeg=30):
        super(CCRNN, self).__init__()
        self.nSeg = nSeg
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=(3-1)//2),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, padding=(3-1)//2),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, padding=(3-1)//2),
            nn.ELU()
        )
        self.conv_fc = nn.Sequential(
            nn.Linear(128*7*5, 1024),
            nn.ELU(),
            nn.Dropout(drop_ratio)
        )
        self.rnn_module = nn.Sequential(
            nn.LSTM(1024, 64, 2, batch_first=True, dropout=drop_ratio)
        )
        self.rnn_fc = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ELU(),
            nn.Dropout(drop_ratio)
        )
        self.readout = nn.Sequential(
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        out = self.conv_module(x)
        out = out.reshape(out.shape[0], np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        out = out.reshape(-1, self.nSeg, out.shape[-1])
        out, (hn, cn) = self.rnn_module(out)
        out = out[:, -1]
        out = self.rnn_fc(out)
        out = self.readout(out)
        return out