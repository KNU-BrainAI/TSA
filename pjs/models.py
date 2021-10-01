import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Causal Dilated
class FBTSANet7(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(FBTSANet7,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(4, False)
        )
        self.layer2 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(4, False)
        )
        # Conv2D Layer2
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(Chans, 1),groups=2),
            nn.BatchNorm2d(8, False),
        )
        self.tcn_block1 = nn.Sequential(
            nn.Conv1d(8, 8, 2, 4, dilation = 3),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Conv1d(8, 8, 3, 3, dilation = 1),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Dropout(0.3)
            )
        
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(1024 , class_num)
        
    def forward(self, x):
        y1 = F.elu(self.layer1(x))
        y2 = F.elu(self.layer2(x))
        y = torch.cat([y1,y2], dim=1)
        # Conv2D
        y = F.elu(self.layer5(y))
       # print(y.size())
        y = torch.squeeze(y, axis=2)
        #print(y.size())
        y = self.tcn_block1(y)
        #print(y.size())
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y
    
class FBTSANet6(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(FBTSANet6,self).__init__()
    
        # Conv2D Layer2
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(Chans, 1),groups=1),
            nn.BatchNorm2d(8, False),
        )
        self.tcn_block1 = nn.Sequential(
            nn.Conv1d(8, 8, 2, 4, dilation = 3),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Conv1d(8, 8, 3, 3, dilation = 1),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Dropout(0.3)
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
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(1024 , class_num)
        
    def stacking(self, f1,f2,f3,index):
        y1 = torch.cat([f1[:,(index-1):index,:,:], f2[:,(index-1):index,:,:]], dim=1)
        y1 = torch.cat([y1[:,:,:,:], f3[:,(index-1):index,:,:]], dim=1)
        return y1
    
    def forward(self, x):
        # Conv2D
        y = self.layer5(x)
       # print(y.size())
        y = torch.squeeze(y, axis=2)
        #print(y.size())
        y = self.tcn_block1(y)
        #print(y.size())
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y

#hilbert
class FBTSANet5(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(FBTSANet5,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        self.layer2 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength//2)-1) //2) +1),(((kernLength//2)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, (kernLength//2)), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        self.layer3 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength//4)-1) //2) +1),(((kernLength//4)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, (kernLength//4)), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1,1)),
            nn.AvgPool2d(kernel_size=(1, 8))
        )
        
        # Conv2D Layer2
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1*D, kernel_size=(Chans, 1),groups=F1),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 1)),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(kernel_size=(1, 16))
        )
        
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(192 , class_num)
        
    def stacking(self, f1,f2,index):
        y1 = torch.cat([f1[:,(index-1):index,:,:], f2[:,(index-1):index,:,:]], dim=1)
       # y1 = torch.cat([y1[:,:,:,:], f3[:,(index-1):index,:,:]], dim=1)
        return y1
    
    def forward(self, x):
        # Conv2D
        f1 = self.layer1(x)
        f2 = self.layer2(x)
        # f size = torch.Size([8, 8, 64, 1536])
       
        y1 = self.stacking(f1,f2,1)
        y1 =  F.elu(self.layer4(y1))
        y2 = self.stacking(f1,f2,2)
        y2 = F.elu(self.layer4(y2))
        y3 = self.stacking(f1,f2,3)
        y3 = F.elu(self.layer4(y3))
        y4 = self.stacking(f1,f2,4)
        y4 = F.elu(self.layer4(y4))
        y5 = self.stacking(f1,f2,5)
        y5 = F.elu(self.layer4(y5))
        y6 = self.stacking(f1,f2,6)
        y6 = F.elu(self.layer4(y6))
        y7 = self.stacking(f1,f2,7)
        y7 = F.elu(self.layer4(y7))
        y8 = self.stacking(f1,f2,8)
        y8 = F.elu(self.layer4(y8))
        
        #print(y1.size())
        s = torch.cat([y1[:,:,:,:], y2[:,:,:,:]], dim=1)
        s = torch.cat([s[:,:,:,:], y3[:,:,:,:]], dim=1)
        s = torch.cat([s[:,:,:,:], y4[:,:,:,:]], dim=1)
        s = torch.cat([s[:,:,:,:], y5[:,:,:,:]], dim=1)
        s = torch.cat([s[:,:,:,:], y6[:,:,:,:]], dim=1)
        s = torch.cat([s[:,:,:,:], y7[:,:,:,:]], dim=1)
        s = torch.cat([s[:,:,:,:], y8[:,:,:,:]], dim=1)
        #print(s.size())
        # Depthwise conv2D
        y = F.elu(self.layer5(s))
        y = F.dropout(y, 0.5)
       
        #print(y.size())
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y
    

#conpcept4
class FBTSANet4(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 32, kernLength = 64, Chans = 64):
        super(FBTSANet4,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
           nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1*D, kernel_size=(Chans, 1),groups=1),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 1)),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(kernel_size=(1, 16))
        )
         # Residual Block
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
       
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(192 , class_num)

    def forward(self, x):
        # Conv2D
        y1 = self.layer1(x)

        # 1x1 Conv2D
        y1 = F.elu(self.layer2(y1))
        y1 = F.dropout(y1, 0.5)
        
       
    
        # Depthwise conv2D
        y = F.elu(self.layer3(y1))
        y = F.dropout(y, 0.5)
       
        
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y

class FBTSANet3(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(FBTSANet3,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
           nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=F1*D, kernel_size=(Chans, 1),groups=2),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 1)),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(kernel_size=(1, 16))
        )
        
        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength)-1) //2) +1),(((kernLength)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, (kernLength)), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
            nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
       
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(192 , class_num)

    def forward(self, x):
        # Conv2D
        y1 = self.layer1(x)
        y2 = self.layer4(x)
        # 1x1 Conv2D
        y1 = F.elu(self.layer2(y1))
        y1 = F.dropout(y1, 0.5)
        
        y2 = F.elu(self.layer5(y2))
        y2 = F.dropout(y2, 0.5)
        
        y = torch.cat([y1,y2], dim=1)
        # Depthwise conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
       
        
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y

class FBTSANet2(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(FBTSANet2,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=8, kernel_size=(1,1)),
           nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=F1*D, kernel_size=(Chans, 1),groups=8),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 1)),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(kernel_size=(1, 16))
        )
        
        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength//2)-1) //2) +1),(((kernLength//2)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, (kernLength//2)), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=8, kernel_size=(1,1)),
            nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
       
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(192 , class_num)

    def forward(self, x):
        # Conv2D
        y1 = self.layer1(x)
        y2 = self.layer4(x)
        # 1x1 Conv2D
        y1 = F.elu(self.layer2(y1))
        y1 = F.dropout(y1, 0.5)
        
        y2 = F.elu(self.layer5(y2))
        y2 = F.dropout(y2, 0.5)
        
        y = torch.cat([y1,y2], dim=1)
        # Depthwise conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
       
        
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y

# Concept3
class EEGNet_3_stacking(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 16, D = 2, F2 = 16, kernLength = 256, Chans = 64):
        super(EEGNet_3_stacking,self).__init__()
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
            nn.Conv2d(in_channels=1, out_channels=F2, kernel_size=(32, 1), groups=1),
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, 1)),
            nn.BatchNorm2d(F2, False),
            nn.AvgPool2d(1, 8)
        )
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(192 , class_num)
    def stacking(self, x):
        out = torch.transpose(x,1,2)
        return out
    def forward(self, x):
        # Conv2D
        y = self.layer1(x)
        y = F.dropout(y, 0.5)
        # Depthwise conv2D
        y = F.elu(self.layer2(y))
        y = F.dropout(y, 0.5)
    
        y = self.stacking(y)
    
        # Separable conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y

class FBTSANet(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(FBTSANet,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
           nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=F1*D, kernel_size=(Chans, 1),groups=2),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 1)),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(kernel_size=(1, 16))
        )
        
        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength//2)-1) //2) +1),(((kernLength//2)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, (kernLength//2)), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
            nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
       
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(192 , class_num)

    def forward(self, x):
        # Conv2D
        y1 = self.layer1(x)
        y2 = self.layer4(x)
        # 1x1 Conv2D
        y1 = F.elu(self.layer2(y1))
        y1 = F.dropout(y1, 0.5)
        
        y2 = F.elu(self.layer5(y2))
        y2 = F.dropout(y2, 0.5)
        
        y = torch.cat([y1,y2], dim=1)
        # Depthwise conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
       
        
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y
    
class FBTSANet_RB(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 16, kernLength = 256, Chans = 64):
        super(FBTSANet_RB,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
           nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=F1*D, kernel_size=(Chans, 1),groups=2),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 1)),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(kernel_size=(1, 16))
        )
        
        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength//2)-1) //2) +1),(((kernLength//2)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, (kernLength//2)), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
            nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Residual Block
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
       
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(192 , class_num)

    def forward(self, x):
        # Conv2D
        y1 = self.layer1(x)
        y2 = self.layer4(x)
        # 1x1 Conv2D
        y1 = F.elu(self.layer2(y1))
        y1 = F.dropout(y1, 0.5)
        
        y2 = F.elu(self.layer5(y2))
        y2 = F.dropout(y2, 0.5)
        
        y = torch.cat([y1,y2], dim=1)
        # Depthwise conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
        # y size = (8,16,1,12)
        y = torch.squeeze(y, axis=2)
        # y size = (8,16,12)
        rb1 = self.tcn_block1(y)
        y = y+rb1
        rb2 = self.tcn_block2(y)
        y = y + rb2
        
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y

class FBTSANet_RB2(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 16, kernLength = 256, Chans = 64):
        super(FBTSANet_RB2,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
           nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=F1*D, kernel_size=(Chans, 1),groups=2),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 1)),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(kernel_size=(1, 16))
        )
        
        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength//2)-1) //2) +1),(((kernLength//2)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, (kernLength//2)), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
            nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Residual Block
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
    
       
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(192 , class_num)

    def forward(self, x):
        # Conv2D
        y1 = self.layer1(x)
        y2 = self.layer4(x)
        # 1x1 Conv2D
        y1 = F.elu(self.layer2(y1))
        y1 = F.dropout(y1, 0.5)
        
        y2 = F.elu(self.layer5(y2))
        y2 = F.dropout(y2, 0.5)
        
        y = torch.cat([y1,y2], dim=1)
        # Depthwise conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
        # y size = (8,16,1,12)
        y = torch.squeeze(y, axis=2)
        # y size = (8,16,12)
        rb1 = self.tcn_block1(y)
        y = y+rb1
        
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y

class FBTSANet_RB3(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 16, kernLength = 256, Chans = 64):
        super(FBTSANet_RB3,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
           nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=F1*D, kernel_size=(Chans, 1),groups=2),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 1)),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(kernel_size=(1, 16))
        )
        
        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength//2)-1) //2) +1),(((kernLength//2)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, (kernLength//2)), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
            nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Residual Block
      
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
       
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(192 , class_num)

    def forward(self, x):
        # Conv2D
        y1 = self.layer1(x)
        y2 = self.layer4(x)
        # 1x1 Conv2D
        y1 = F.elu(self.layer2(y1))
        y1 = F.dropout(y1, 0.5)
        
        y2 = F.elu(self.layer5(y2))
        y2 = F.dropout(y2, 0.5)
        
        y = torch.cat([y1,y2], dim=1)
        # Depthwise conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
        # y size = (8,16,1,12)
        y = torch.squeeze(y, axis=2)
        # y size = (8,16,12)
    
        rb2 = self.tcn_block2(y)
        y = y + rb2
        
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y

class FBTSANet_RB4(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 16, kernLength = 256, Chans = 64):
        super(FBTSANet_RB4,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
           nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Conv2D Layer2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=F1*D, kernel_size=(Chans, 1),groups=2),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 1)),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(kernel_size=(1, 16))
        )
        
        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(( ( ( ((kernLength//2)-1) //2) +1),(((kernLength//2)-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, (kernLength//2)), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1)),
            nn.AvgPool2d(kernel_size=(1, 8))
        )
        # Residual Block
      
        self.tcn_block2 = nn.Sequential(
            nn.ZeroPad2d((3,3,0,0)),
            nn.Conv1d(F2, F2, 4, 1, dilation=1),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.ZeroPad2d((3,3,0,0)),
            nn.Conv1d(F2, F2, 4, 1, dilation=2),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),
            )
       
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(240 , class_num)

    def forward(self, x):
        # Conv2D
        y1 = self.layer1(x)
        y2 = self.layer4(x)
        # 1x1 Conv2D
        y1 = F.elu(self.layer2(y1))
        y1 = F.dropout(y1, 0.5)
        
        y2 = F.elu(self.layer5(y2))
        y2 = F.dropout(y2, 0.5)
        
        y = torch.cat([y1,y2], dim=1)
        # Depthwise conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
        # y size = (8,16,1,12)
        y = torch.squeeze(y, axis=2)
        # y size = (8,16,12)
    
        rb2 = self.tcn_block2(y)
        y =  rb2
        
        
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y
# Concept2
class EEGAtteNet(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D=2, F2 = 16, kernLength = 256, Chans = 64):
        super(EEGAtteNet,self).__init__()
     
        self.Wk =  nn.Linear(128,128)
        self.Wv =  nn.Linear(8,8)
        self.ff1 = nn.Conv1d(4,8,kernel_size=1)
        self.ff2 = nn.Conv1d(8,4,kernel_size=1)
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False),
            nn.AvgPool2d((1, 4),stride=(1,4)),
        )
        # Depthwise Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1*D, kernel_size=(Chans, 1), groups=F1),
            nn.BatchNorm2d(F1*D, False),
            nn.GELU(),
            nn.Dropout(0.8),
        )
        # Separable Layer
        self.layer3 = nn.Sequential(
            #Padding size = 1st Conv2D Filter(W,H) -> Pad((H-1)/2,(H-1)/2,(W-1)/2,(W-1)/2)
            nn.ZeroPad2d(((((32 - 1) // 2 ) + 1), ( (32 - 1) // 2), 0, 0)),
            nn.Conv2d(in_channels=F1*D, out_channels=F2, kernel_size=(1, 32), groups=1),
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, 1)),
            nn.BatchNorm2d(F2, False),
            nn.GELU(),
            nn.AvgPool2d((1, 8), stride = (1,8)),
            nn.Dropout(0.8),
        )
        # Flatten (1x32x16)
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(512 , class_num)
                
  
    def Attention_score(self,x):
        #K = self.Wk(x)
        #print(torch.transpose(x,2,3).size())
        #print("Ksize:",K.size())
        #print("xsize:",x.size())
        Att_score = torch.matmul(x,torch.transpose(x,2,3))
        Att_score = Att_score.mul_(1/4**0.5)
        #print("ATT size:",Att_score.size())
        return Att_score
    
    def forward(self, x):
        y = self.layer1(x)
        Q = y
        #print("Q size:",Q.size())
        As = self.Attention_score(Q)
        As = nn.Softmax(dim=-1)(As)
        y = torch.matmul(As,Q)
        #print("R size:",y.size())
        y = self.layer2(y)
        y = self.layer3(y)
        #print("2 y size:",y.size())
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y

class TSAtteNet(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, F2 = 8, kernLength = 256, Chans = 64):
        super(TSAtteNet,self).__init__()
     
        self.Wk =  nn.Linear(128,128)
        self.ff1 = nn.Conv1d(4,8,kernel_size=1)
        self.ff2 = nn.Conv1d(8,4,kernel_size=1)
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1), groups=F1)
        )
        # Conv2D Layer2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F2, kernel_size=(Chans, 1),groups=1),
            nn.BatchNorm2d(F2, False),
            nn.AvgPool2d(1, 4)
        )
        # Flatten (1x32x16)
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(256 , class_num)
    def Divide_word(self,x):
        #input shape = (16,16,1,32) [batchsize,chan,Row,Col]
        word = []
        word1 = x[:,:,:,0:8]
        word2 = x[:,:,:,8:16]
        word3 = x[:,:,:,16:24]
        word4 = x[:,:,:,24:32]
        word = torch.cat([word1,word2,word3,word4],dim=2)
        return word
                         
  
    def Attention_score(self,x):
        #K = self.Wk(x)
        #print(torch.transpose(x,2,3).size())
        #print("Ksize:",K.size())
        #print("xsize:",x.size())
        Att_score = torch.matmul(x,torch.transpose(x,2,3))
        Att_score = Att_score.mul_(1/4**0.5)
        #print("ATT size:",Att_score.size())
        return Att_score
    
    def forward(self, x):
        # Conv2D
        y = self.layer1(x)
        # 1x1 Conv2D
        y = self.layer2(y)
        # Depthwise conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
        #print(y.size())
        #Attention
        temp = self.Divide_word(y)
        #print(temp.size())
        #y = torch.cat([temp[:,0,:,:],temp[:,1,:,:],temp[:,2,:,:],temp[:,3,:,:],temp[:,4,:,:],temp[:,5,:,:],temp[:,6,:,:],
        #            temp[:,7,:,:],temp[:,8,:,:],temp[:,9,:,:],temp[:,10,:,:],temp[:,11,:,:],temp[:,12,:,:],temp[:,13,:,:],
        #             temp[:,14,:,:],temp[:,15,:,:]],dim = 2)
       
        
        #print(temp.size())
        Q = temp
        #print("Q size:",Q.size())
        As = self.Attention_score(Q)

        As = nn.Softmax(dim=-1)(As)

        R = torch.matmul(As,Q)
        #print("R size:",R.size())
        #y = self.ff1(R)
        #y = F.gelu(y)
        #y = self.ff2(y)
        #print("y size:",y.size())
        #print(self.W1)
        # Flatten
        y = self.flatten(R)
        # Linear
        y = self.linear1(y)
        return y    

class EEGAtteNet2(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D=2, F2 = 16, kernLength = 256, Chans = 64):
        super(EEGAtteNet2,self).__init__()
        self.Wq =  nn.Linear(32,32)
        self.Wk =  nn.Linear(32,32)
        self.Wv =  nn.Linear(32,32)
        self.ff1 = nn.Conv1d(4,8,kernel_size=1)
        self.ff2 = nn.Conv1d(8,4,kernel_size=1)
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
            nn.ELU(),
            nn.AvgPool2d(1, 32),
            nn.Dropout(0.8),
        )
        # Separable Layer
        self.layer3 = nn.Sequential(
            #Padding size = 1st Conv2D Filter(W,H) -> Pad((H-1)/2,(H-1)/2,(W-1)/2,(W-1)/2)
            nn.ZeroPad2d(((((32 - 1) // 2 ) + 1), ( (32 - 1) // 2), 0, 0)),
            nn.Conv2d(in_channels=F1*D, out_channels=F2, kernel_size=(1, 32), groups=F2),
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, 1),groups=F2),
            nn.BatchNorm2d(F2, False),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride = (1,8)),
            nn.Dropout(0.8),
        )
        # Flatten (1x32x16)
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(512 , class_num)
    def Divide_word(self,x):
        #input shape = (16,16,1,32) [batchsize,chan,Row,Col]
        word = []
        word1 = x[:,:,:,0:8]
        word2 = x[:,:,:,8:16]
        word3 = x[:,:,:,16:24]
        word4 = x[:,:,:,24:32]
        word = torch.cat([word1,word2,word3,word4],dim=2)
        return word
                         
  
    def Attention_score(self,x):
        Q = self.Wq(x)
        K = self.Wk(x)
        #print(torch.transpose(x,2,3).size())
       #print("Ksize:",K.size())
       # print("xsize:",x.size())
        Att_score = torch.matmul(Q,torch.transpose(K,2,3))
        Att_score = Att_score.mul_(1/16**0.5)
        #print("ATT size:",Att_score.size())
        return Att_score
    
    def forward(self, x):
        # Conv2D
        y = self.layer1(x)
        # 1x1 Conv2D
        y = self.layer2(y)
        y = torch.transpose(y, 1, 2)
        #print(y.size())
        # Depthwise conv2D
        #y = self.layer3(y)
        #print(y.size())
        #Attention
        #y = self.Divide_word(y)
        #print(temp.size())
        #y = torch.cat([temp[:,0,:,:],temp[:,1,:,:],temp[:,2,:,:],temp[:,3,:,:],temp[:,4,:,:],temp[:,5,:,:],temp[:,6,:,:],
        #            temp[:,7,:,:],temp[:,8,:,:],temp[:,9,:,:],temp[:,10,:,:],temp[:,11,:,:],temp[:,12,:,:],temp[:,13,:,:],
        #             temp[:,14,:,:],temp[:,15,:,:]],dim = 2)
       
        
        #print(temp.size())
        
        #print("Q size:",Q.size())
        As = self.Attention_score(y)

        As = nn.Softmax(dim=-1)(As)
        V = self.Wv(y)
        #print(y.size())
        y = torch.matmul(As,V)
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y

class EEGNet_2_ch_weight(nn.Module):
    
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    
    def __init__(self, bias = False, class_num = 2, F1 = 16, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(EEGNet_2_ch_weight,self).__init__()
             
        self.ch_weight =  nn.Parameter(torch.tensor(torch.rand((64,1), requires_grad = True)))
        
        
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
        self.linear1 = nn.Linear(F2 * 2 , class_num)
        
       
    def forward(self, x):
        tmp = x * self.ch_weight
        y = x + tmp
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
    
class TSANet(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 8, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(TSANet,self).__init__()
        # Conv2D Layer
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
        )
        #1x1Conv2D Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=1, kernel_size=(1,1))
        )
        # Conv2D Layer2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1*D, kernel_size=(Chans, 1),groups=1),
            nn.BatchNorm2d(F1*D, False),
            nn.AvgPool2d(1, 32)
        )
        # Flatten
        self.flatten = nn.Flatten()

        # Linear
        self.linear1 = nn.Linear(512 , class_num)

    def forward(self, x):
        # Conv2D
        y = self.layer1(x)
        # 1x1 Conv2D
        y = self.layer2(y)
        # Depthwise conv2D
        y = F.elu(self.layer3(y))
        y = F.dropout(y, 0.5)
        # Flatten
        y = self.flatten(y)
        # Linear
        y = self.linear1(y)
        return y
# Concept1
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

class EEGNet_overlap(nn.Module):
    # kernelLength = sampling_rate/2 , Chans = EEG channel num
    def __init__(self, bias = False, class_num = 2, F1 = 16, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(EEGNet_overlap,self).__init__()
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
        self.linear1 = nn.Linear(F2 * 2 , class_num)

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

#Skip Connection
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
    

# Paper EEGNet
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
    
# TSA EEGNet
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
            nn.Conv2d(25, 25, (64, 1), 1, bias=False),
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
    def __init__(self, bias = False, class_num = 2, F1 = 16, D = 2, F2 = 32, kernLength = 256, Chans = 64):
        super(EEG_TCNet, self).__init__()
        F2 = F1*D

        self.conv_temporal = nn.Sequential(
            nn.ZeroPad2d(( ( ( (kernLength-1) //2) +1),((kernLength-1)//2 ),0,0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernLength), groups=1,bias=bias),
            nn.BatchNorm2d(F1, False)
            )

        self.conv_spatial = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1*D, kernel_size=(Chans, 1), groups=F1),
            nn.BatchNorm2d(F1*D, False),
            nn.ELU(),
            nn.AvgPool2d((1,16)),
            nn.Dropout(0.5)
            )

        self.conv_separable = nn.Sequential(
            nn.ZeroPad2d(((((32 - 1) // 2 ) + 1), ( (32 - 1) // 2), 0, 0)),
            nn.Conv2d(in_channels=F1*D, out_channels=F2, kernel_size=(1, 32), groups=F1*D),
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, 1)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,32)), #(12)
            nn.Dropout(0.5)
            )

        self.conv_fc = nn.Sequential(
            ConstrainedLinear(96, class_num)
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