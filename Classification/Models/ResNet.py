import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, bottleneck = False, preact=False, in_stride = 1, projection='B'):
        super(ResidualBlock, self).__init__()
        
        # define residual
        if (preact):
            self.layers0 = nn.Sequential(
                nn.BatchNorm2d(input_channels),
                nn.ReLU(inplace=True)
                )
        else:
            self.layers0 = nn.Sequential()

        self.preact = preact

        if projection == 'D':
            stride1 = 1
            stride2 = in_stride
        else:
            stride1 = in_stride
            stride2 = 1

        layers = []
        if bottleneck:
            layers.append(nn.Conv2d(input_channels,output_channels//4,1, stride = stride1, bias = False))      
            layers.append(nn.BatchNorm2d(output_channels//4))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(output_channels//4,output_channels//4,3, padding = 1, stride = stride2, bias = False))      
            layers.append(nn.BatchNorm2d(output_channels//4))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(output_channels//4,output_channels,1, bias = False))
        else:
            layers.append(nn.Conv2d(input_channels,output_channels,3, padding = 1, stride = stride1, bias = False))      
            layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(output_channels,output_channels,3, padding = 1, stride = stride2, bias = False))

        if not preact :
            layers.append(nn.BatchNorm2d(output_channels))

        self.layers = nn.Sequential(*layers)

        # define projection
        layers = []
        if in_stride != 1:
            if projection == 'B':
                layers.append(nn.AvgPool2d(2))
            elif projection == 'C':
                layers.append(nn.Conv2d(input_channels,output_channels,1, padding = 0, groups = input_groups, stride = in_stride, bias = False))
                if not preact:
                    layers.append(nn.BatchNorm2d(output_channels))
            else projection == 'D':
                layers.append(nn.AvgPool2d(2))
                layers.append(nn.Conv2d(input_channels,output_channels,1, padding = 0, groups = input_groups, stride = in_stride, bias = False))
                if not preact:
                    layers.append(nn.BatchNorm2d(output_channels))

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
        out = out + x
        if not self.preact:
            return F.relu(out)
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, classes = 10, blocks = [3,3,3], width=16, bottleneck = False, preact=False, projection='B', stem='ImageNet'):
        super(ResNet, self).__init__()

        layers = []
        if dataset == 'ImageNet':
            if projection == 'D':
                current_width = 3
                for i in range(3):
                    layers.append(nn.Conv2d(current_width,width,3,stride=2,padding=1,bias=False))
                    current_width = width
                    if i < 2 or preact == False :
                        layers.append(nn.BatchNorm2d(width))
                        layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Conv2d(3,width,7,stride=2,padding=3,bias=False))
                layers.append(nn.BatchNorm2d(width))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.MaxPool2(3,2,padding=1))

            if bottleneck:
                width = 4*width
        else:
            layers.append(nn.Conv2d(3,width,3,padding=1,bias=False))
            if not preact:
                layers.append(nn.BatchNorm2d(width))
                layers.append(nn.ReLU(inplace=True))
        
        stride = 1
        for block in blocks:
            for n in range(block):
                if (stride == 2):
                    width = 2*width
                    layers.append(ResidualBlock(current_width,width,bottleneck=bottleneck,pract=preact,in_stride=stride,projection=projection))
                    stride = 1
                    current_width = width
                else:
                    layers.append(ResidualBlock(width,width,bottleneck=bottleneck,pract=preact,in_stride=stride,projection=projection))
            stride = 2

        if preact:
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.layers = nn.Sequential(*layers)

        self.width = width
        self.classifier = nn.Linear(width, classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1,self.width)

        return self.classifier(x)


# CIFAR models
def ResNet110(classes = 10, preact=False, projection = 'B'):
    return ResNet(classes=classes, blocks = [18,18,18], width=16, preact=preact, projection=projection, stem='cifar')

def ResNet164(classes = 10, preact=False, projection = 'B'):
    return ResNet(classes=classes, blocks = [18,18,18], width=64, bottleneck = True, preact=preact, projection=projection, stem='cifar')

def ResNet1001(classes = 10, preact=True, projection = 'B'):
    return ResNet(classes=classes, blocks = [111,111,111], width=64, bottleneck = True, preact=preact, projection=projection, stem='cifar')

# ImageNet models
def ResNet18(classes=1000, preact=False, projection='B'):
    return ResNet(classes=classes, blocks = [2, 2, 2, 2], width=64, bottleneck=False, preact=preact, projection=projection)

def ResNet34(classes=1000, preact=False, projection='B'):
    return ResNet(classes=classes, blocks = [3, 4, 6, 3], width=64, bottleneck=False, preact=preact, projection=projection)

def ResNet50(classes=1000, preact=False, projection='B'):
    return ResNet(classes=classes, blocks = [3, 4, 6, 3], width=64, bottleneck=True, preact=preact, projection=projection)

def ResNet101(classes=1000, preact=False, projection='B'):
    return ResNet(classes=classes, blocks = [3, 4, 23, 3], width=64, bottleneck=True, preact=preact, projection=projection)

def ResNet152(classes=1000, preact=False, projection='B'):
    return ResNet(classes=classes, blocks = [3, 8, 36, 3], width=64, bottleneck=True, preact=preact, projection=projection)

def ResNet200(classes=1000, preact=True, projection='B'):
    return ResNet(classes=classes, blocks = [3, 24, 36, 3], width=64, bottleneck=True, preact=preact, projection=projection)
