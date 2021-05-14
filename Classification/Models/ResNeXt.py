import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, cardinality, inner_width, in_stride = 1, projection='B'):
        super(ResidualBlock, self).__init__()
        
        # define residual
        self.layers0 = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
            )


        in_width = cardinality*inner_width
        layers = []
        layers.append(nn.Conv2d(input_channels, in_width,1, stride = in_stride, bias = False))      
        layers.append(nn.BatchNorm2d(in_width))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_width,in_width,3, padding = 1, groups = cardinality, bias = False))      
        layers.append(nn.BatchNorm2d(in_width))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_width,output_channels,1, bias = False))


        self.layers = nn.Sequential(*layers)

        # define projection
        layers = []
        if in_stride != 1:
            if projection == 'C':
                layers.append(nn.Conv2d(input_channels,output_channels,1, padding = 0, groups = input_groups, stride = in_stride, bias = False))
            else:
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


class ResNeXt(nn.Module):
    def __init__(self, classes = 10, blocks = [3,3,3], width=64, cardinality=32, in_width = 4,   projection='B', stem='ImageNet'):
        super(ResNeXt, self).__init__()

        layers = []
        if dataset == 'ImageNet':
            layers.append(nn.Conv2d(3,width,7,stride=2,padding=3,bias=False))
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2(3,2,padding=1))
            width = 4*width
        else:
            layers.append(nn.Conv2d(3,width,3,padding=1,bias=False))
        
        stride = 1
        for block in blocks:
            for n in range(block):
                if (stride == 2):
                    width = 2*width
                    in_width = 2*in_width
                    layers.append(ResidualBlock(current_width, width, cardinality, in_width, in_stride=stride, projection=projection))
                    stride = 1
                    current_width = width
                else:
                    layers.append(ResidualBlock(width, width, cardinality, in_width, in_stride=stride, projection=projection))
            stride = 2

        layers.append(nn.BatchNorm2d(width))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.layers = nn.Sequential(*layers)


        self.classifier = nn.Linear(width, classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1,self.width)

        return self.classifier(x)

def ResNeXt50_32_4(classes=1000, preact=False, projection='B'):
    return ResNeXt(classes=classes,blocks = [3, 4, 6, 3], width=64, cardinality = 32, in_width = 4, projection=projection)

def ResNeXt101_32_4(classes=1000, preact=False, projection='B'):
    return ResNeXt(classes=classes,blocks = [3, 4, 23, 3], width=64, cardinality = 32, in_width = 4, projection=projection)

def ResNet152_32_4(classes=1000, preact=False, projection='B'):
    return ResNeXt(classes=classes,blocks = [3, 8, 36, 3], width=64, cardinality = 32, in_width = 4, projection=projection)

def ResNeXt29_32_4(classes = 10, preact=False, projection = 'B'):
    return ResNet(classes=classes,blocks = [3,3,3], width=4, cardinality = 32, in_width = 4, projection = projection, stem='cifar')