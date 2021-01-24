import torch
import torch.nn as nn

class Inception(nn.Module):
    '''
    Inception block for a GoogLeNet-like CNN

    Attributes
    ----------
    in_planes : int
        Number of input feature maps
    n1x1 : int
        Number of direct 1x1 convolutions
    n3x3red : int
        Number of 1x1 reductions before the 3x3 convolutions
    n3x3 : int
        Number of 3x3 convolutions
    n5x5red : int
        Number of 1x1 reductions before the 5x5 convolutions
    n5x5 : int
        Number of 5x5 convolutions
    pool_planes : int
        Number of 1x1 convolutions after 3x3 max pooling
    b1 : Sequential
        First branch (direct 1x1 convolutions)
    b2 : Sequential
        Second branch (reduction then 3x3 convolutions)
    b3 : Sequential
        Third branch (reduction then 5x5 convolutions)
    b4 : Sequential
        Fourth branch (max pooling then reduction)
    '''
    
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, withAux = False):
        super(Inception, self).__init__()
        self.in_planes = in_planes
        self.n1x1 = n1x1
        self.n3x3red = n3x3red
        self.n3x3 = n3x3
        self.n5x5red = n5x5red
        self.n5x5 = n5x5
        self.pool_planes = pool_planes
        
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)

class Aux4A(nn.Module):
    def __init__(self):
        super(Aux4A, self).__init__()
        self.conv = nn.Sequential(
            # torch.Size([4, 512, 14, 14])
            # An average pooling layer with 5×5 filter size and stride 3, 
            # resulting in an 4×4×512 output for the (4a)
            nn.AvgPool2d(kernel_size=5,stride=3,padding=0),
            # A 1×1 convolution with 128 filters for dimension reduction and rectified linear activation
            nn.Conv2d(in_channels=512,out_channels=128,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
            # torch.Size([4, 128, 4, 4])
        )
        self.fc = nn.Sequential(
            # torch.Size([4, 128, 4, 4])
            # A fully connected layer with 1024 units and rectified linear activation.
            nn.Linear(in_features=128 * 4 * 4,out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024,out_features=1024),
            nn.ReLU(inplace=True),
            # A dropout layer with 70% ratio of dropped outputs.
            nn.Dropout2d(p=0.7),
            # A linear layer with softmax loss as the classifier (predicting the same 1000 classes as the main classifier, but removed at inference time).
            nn.Linear(in_features=1024,out_features=10),
            # nn.Softmax()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out.flatten(start_dim=1))
        return out


class Aux4D(nn.Module):
    def __init__(self):
        super(Aux4D, self).__init__()
        self.conv = nn.Sequential(
            # torch.Size([4, 528, 14, 14])
            # An average pooling layer with 5×5 filter size and stride 3, 
            # resulting in an 4×4×528 output for the (4d)
            nn.AvgPool2d(kernel_size=5,stride=3,padding=0),
            # A 1×1 convolution with 128 filters for dimension reduction and rectified linear activation
            nn.Conv2d(in_channels=528,out_channels=128,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
            # torch.Size([4, 128, 4, 4])
        )
        self.fc = nn.Sequential(
            # torch.Size([4, 128, 4, 4])
            # A fully connected layer with 1024 units and rectified linear activation.
            nn.Linear(in_features=128 * 4 * 4,out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024,out_features=1024),
            nn.ReLU(inplace=True),
            # A dropout layer with 70% ratio of dropped outputs.
            nn.Dropout2d(p=0.7),
            # A linear layer with softmax loss as the classifier (predicting the same 1000 classes as the main classifier, but removed at inference time).
            nn.Linear(in_features=1024,out_features=10),
            # nn.Softmax()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out.flatten(start_dim=1))
        return out

class PreLayer(nn.Module):
    def __init__(self):
        super(PreLayer, self).__init__()
        self.pre_layers = nn.Sequential(
            # nn.Conv2d(3, 192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            # nn.ReLU(True),

            # Follow GoogLeNet Implementation
            # Note: remove BatchNorm2d because it is not used in the paper
            # 1. Conv 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=7,stride=2,padding=3),
            nn.ReLU(inplace=True),
            # 2. MaxPool 112x112x64 -> 56x56x64
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            # 3. Conv 56x56x64 -> 56x56x192
            ### 3x3red
            nn.Conv2d(64, 64, kernel_size=1),
            ### 3x3 
            nn.Conv2d(64,192, kernel_size=3,stride=1,padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            # 4. MaxPool 56x56x192 -> 28x28x192
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

    def forward(self, x):
        out = self.pre_layers(x)
        return out

class GoogLeNet(nn.Module):

    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = PreLayer()

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        # torch.Size([4, 512, 14, 14])
        self.aux4a = Aux4A()
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        
        self.aux4d = Aux4D()
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        # self.linear = nn.Linear(1024, 10)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024,out_features=10),
            # nn.Softmax()
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)

        out_aux4a = self.aux4a(out)
        out = self.b4(out)
        
        out = self.c4(out)
        out = self.d4(out)
        # print("d4", out.shape)

        out_aux4d = self.aux4d(out)
        out = self.e4(out)

        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        # print(out.shape)
        out = self.fc(out.flatten(start_dim=1))
        # print("out",out.shape)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        
        if(self.training):
            return out, out_aux4a, out_aux4d
        else:
            return out