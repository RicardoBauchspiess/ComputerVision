import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, bottleneck = False, in_stride = 1):
        super(ResidualBlock, self).__init__()
        
        # define residual
        self.layers0 = nn.BatchNorm2d(input_channels)

        layers = []
        if bottleneck:
            layers.append(nn.Conv2d(input_channels,output_channels//4,1, stride = in_stride, bias = False))      
            layers.append(nn.BatchNorm2d(output_channels//4))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(output_channels//4,output_channels//4,3, padding = 1, bias = False))      
            layers.append(nn.BatchNorm2d(output_channels//4))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(output_channels//4,output_channels,1, bias = False))
        else:
            layers.append(nn.Conv2d(input_channels,output_channels,3, padding = 1, stride = in_stride, bias = False))      
            layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(output_channels,output_channels,3, padding = 1, bias = False))

        layers.append(nn.BatchNorm2d(output_channels))

        self.layers = nn.Sequential(*layers)

        # define projection
        layers = []
        if in_stride != 1:
            layers.append(nn.AvgPool2d(2))

        self.skip = nn.Sequential(*layers)
                    
            

    def forward(self, x):
        x = self.layers0(x)

        out = self.layers(x)
        x = self.sequential(x)

        b,c,h,w = x.size()
        _,c2,_,_ = out.size()

        if (c != c2):
            padding = torch.autograd.Variable(torch.FloatTensor(b, (c2-c) , w, h).fill_(0))
            padding.to(device=x.device)
            x = torch.cat([x, padding], dim=1)
        return out + x

class PyramidNet(nn.Module):
    def __init__(self, classes = 10, blocks = [3,3,3], width=16, alpha = 48, bottleneck = False, stem='ImageNet'):
        super(PyramidNet, self).__init__()

        

        #bottleneck aspect ratio
        r = 1 if not bottleneck else 4

        # variables to control width growth
        width0 = width
        total_blocks = sum(blocks)
        pos = 0

        layers = []
        if dataset == 'ImageNet':
            layers.append(nn.Conv2d(3,width,7,stride=2,padding=3,bias=False))
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2(3,2,padding=1))
        else:
            layers.append(nn.Conv2d(3,width,3,padding=1,bias=False))
        
        current_width = width
        stride = 1
        for block in blocks:
            for n in range(block):
                pos += 1
                width = width0 + (pos*alpha)//total_blocks
                if (stride == 2):
                    layers.append(ResidualBlock(current_width,width*r,bottleneck=bottleneck,pract=preact,in_stride=stride,projection=projection))
                    stride = 1
                else:
                    layers.append(ResidualBlock(current_width,width*r,bottleneck=bottleneck,pract=preact,in_stride=stride,projection=projection))
                current_width = width*r
            stride = 2

        layers.append(nn.BatchNorm2d(width*r))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.layers = nn.Sequential(*layers)


        self.classifier = nn.Linear(width*r, classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1,self.width)

        return self.classifier(x)

def PyramidNet110(classes = 10, alpha = 48):
    return PyramidNet(classes=classes, blocks = [18,18,18], width=16, alpha = alpha, stem='cifar')

def PyramidNet200(classes = 1000, alpha = 300):
    return PyramidNet(classes=classes, blocks = [3,24,36,3], width=64, bottlenck = True, alpha = alpha)

