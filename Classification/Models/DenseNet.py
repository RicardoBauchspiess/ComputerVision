import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, input_channels, output_channels, BC = True):
        super(DenseLayer, self).__init__()
        

        layers = []
        if BC:
            layers.append(nn.BatchNorm2d(input_channels))
            layers.append(nn.ReLU(inplace = True))
            layers.append(nn.Conv2d(input_channels,output_channels*4,1, padding = 0, bias = False))      
            layers.append(nn.BatchNorm2d(output_channels*4))
            layers.append(nn.ReLU(inplace = True))
            layers.append(nn.Conv2d(output_channels*4,output_channels,3, padding = 1, bias = False))
        else:
            layers.append(nn.BatchNorm2d(input_channels))
            layers.append(nn.ReLU(inplace = True))
            layers.append(nn.Conv2d(input_channels,output_channels,3, padding = 1, bias = False))
        
        

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(torch.cat(x,dim=1))

class Transition(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Transition, self).__init__()
        
        layers = []
        
        layers.append(nn.BatchNorm2d(input_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(input_channels,output_channels,1, bias = False))     

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return [F.avg_pool2d(self.layers(torch.cat(x,dim=1 )),2)]

class DenseBlock(nn.Module):
    def __init__(self, input_width, width, layers, BC = True):
        super(DenseBlock, self).__init__()
        self.dense = nn.ModuleList([DenseLayer(input_width+i*width, width, BC = BC) for i  in range(layers) ])
    def forward(self, x):
        for i in range(self.layers):
            x.append(self.dense[i](x))
        return x


class DenseNet(nn.Module):
    def __init__(self, classes = 10, blocks = [16,16,16], BC = True, width = 12, stem = 'ImageNet'):
        super(DenseNet, self).__init__()


        layers = []
        if dataset == 'ImageNet':
            layers.append(nn.Conv2d(3,2*width,7,stride=2,padding=3,bias=False))
            layers.append(nn.BatchNorm2d(2*width))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2(3,2,padding=1))
        else:
            layers.append(nn.Conv2d(3,2*width,3,padding=1,bias=False))
        self.stem = nn.Sequential(*layers)


        layers = []

        current_width = 2*width
        i = 0
        for block in blocks:
            layers.append(DenseBlock(current_width,width,block, BC))
            i += 1
            current_width += width*block
            if i < len(blocks):
                if BC:
                    layers.append(Transition(current_width, current_width//2, BC))
                    current_width = current_width//2
                else:
                    layers.append(Transition(current_width, current_width, BC))

        self.layers = nn.Sequential(*layers)

        layers = []

        layers.append(nn.BatchNorm2d(current_width))
        layers.append(nn.ReLU(inplace=True))   
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.end = nn.Sequential(*layers)

        self.width = current_width

        self.classifier = nn.Linear(current_width, classes)


        
    def forward(self, x):
        x = [self.stem(x)]
        x = self.layers(x)
        x = self.end(torch.cat(x,dim=1))
        x = x.view(-1,self.width)
        return self.classifier(x)


# CIFAR, SVHN models
def DenseNet40(classes = 10, width = 12):
    return DenseNet(classes = classes, blocks = [12,12,12], width = width, BC = False, stem = 'cifar')

def DenseNet100(classes = 10, width = 12):
    return DenseNet(classes = classes, blocks = [32,32,32], width = width, BC = False, stem = 'cifar')

def DenseNetBC100(classes = 10, width = 12):
    return DenseNet(classes = classes, blocks = [16,16,16], width = width, BC = True, stem = 'cifar')

def DenseNetBC250(classes = 10, width = 24):
    return DenseNet(classes = classes, blocks = [41,41,41], width = width, BC = True, stem = 'cifar')

def DenseNetBC190(classes = 10, width = 48):
    return DenseNet(classes = classes, blocks = [31,31,31], width = width, BC = True, stem = 'cifar')

# ImageNet models
def DenseNet121(classes = 1000, width = 32):
    return DenseNet(classes = classes, blocks = [6,12,24,16], width = width, BC = True, stem = 'ImageNet')

def DenseNet169(classes = 1000, width = 32):
    return DenseNet(classes = classes, blocks = [6,12,32,32], width = width, BC = True, stem = 'ImageNet')

def DenseNet201(classes = 1000, width = 32):
    return DenseNet(classes = classes, blocks = [6,12,48,32], width = width, BC = True, stem = 'ImageNet')

def DenseNet264(classes = 1000, width = 32):
    return DenseNet(classes = classes, blocks = [6,12,64,48], width = width, BC = True, stem = 'ImageNet')