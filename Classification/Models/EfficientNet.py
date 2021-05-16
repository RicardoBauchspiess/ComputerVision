import torch
import torch.nn as nn
import torch.nn.functional as F


# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze-and-Excitation layer
class SE(nn.Module):
    def __init__(self, channels, in_width):
        super(SE, self).__init__()
        self.fc = nn.Sequential(
        	nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, in_width, 1, padding = 0, bias=False),
            Swish(),
            nn.Conv2d(in_width, channels, 1, padding= 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(y)
        return x * y.expand_as(x)

# Mobile inverted Bottleneck Block + SE layer
class MBConvBlock(nn.Module):
    def __init__(self, in_width, m_width, se_width, out_width, kernel, in_stride=1):
        super().__init__()


        layers = []
        if (in_width != m_width):
            layers.append(nn.Conv2d(in_width, m_width, 1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(m_width))
            layers.append(Swish())
        layers.append(nn.Conv2d(m_width, m_width, kernel, stride=in_stride, padding=(kernel-1)//2, bias=False  ))
        layers.append(nn.BatchNorm2d(m_width))
        layers.append(MemoryEfficientSwish())
        if (se_width>0):
            layers.append(SE(m_width, se_width, out_groups))
        layers.append(nn.Conv2d(m_width, out_width, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_width))

        self.layers = nn.Sequential(*layers)

        if ( in_stride == 1 and in_width == out_width ):
            self.skip = True
        else:
            self.skip = False

    def forward(self, inputs):
        x = self.layers(inputs)
        if (self.skip):
            x = x + inputs
        return x

# round number of filters (multiple of 8)
def get_width(base_width, gain):
	divisor = 8

	width = base_width*gain
	new_width = (int(width + divisor/2) // divisor) * divisor
	new_width = max(divisor, new_width)
	if new_width < 0.9 * width:
		new_width += divisor

	return int(new_width)

# EfficientNet
class EfficientNet(nn.Module):
    def __init__(self, classes, base_width = [32, 16, 24, 40, 80, 112, 192, 320, 1280], base_repeats = [1, 2, 2, 3, 3, 4, 1],
    	kernels = [3, 3, 3, 5, 3, 5, 5, 3, 1], SEratio = 0.25, MBratio = 6, width_gain = 1, depth_gain = 1, dropout = 0.2 ):
        super(EfficientNet, self).__init__()

        layers = []
       
        width = get_width(base_width[0],width_gain)
        kernel = kernels[0]
        pad = (kernel-1)//2

        layers.append(nn.Conv2d(3, width, kernel, padding = pad, stride = 2, bias = False))
        layers.append(nn.BatchNorm2d(width))

        current_width = width

        # first set of MB layers has MBratio = 1
        mbratio = 1
        stride = 1
        for i in range(len(base_repeats)):

        	repeat = int(math.ceil(base_repeats[i]*depth_gain))
        	width = get_width(base_width[i+1],width_gain)
        	kernel = kernels[i+1]

        	for j in range(repeats):
        		
        		SEwidth = int(current_width*SEratio)
        		MBwidth = int(current_width*mbratio)
        	
        		layers.append(current_width, MBwidth, SEwidth, width, kernel, in_stride = stride)
        		current_width = width
        		mbratio = MBratio
        	
        	stride = 2

        width = get_width(base_width[len(base_width)-1],width_gain)
        kernel = kernels[len(kernels)-1]
        pad = (kernel-1)//2

        layers.append(nn.Conv2d(current_width, width, kernel, padding = pad, stride = 1, bias = False))
        layers.append(nn.BatchNorm2d(width))
       	layers.append(Swish())
       	layers.append(nn.AdaptiveAvgPool2d(1))
       	if dropout > 0:
       		layers.append(Dropout(dropout))

       	self.layers = nn.Sequential(*layers)

       	self.classifier = nn.Linear(width, classes)
        	
       	self.width = width

    

    def forward(self, x):
    	x = self.layers(x)
    	x = x.view(-1, self.width)
    	return self.classifier(x)

# Models

# resolution = 224x224
def EfficientNetB0():
	return EfficientNet()

# resolution = 240x240
def EfficientNetB1():
	return EfficientNet(depth_gain = 1.1)

# resolution = 260x260
def EfficientNetB2():
	return EfficientNet(width_gain = 1.1, depth_gain = 1.2, dropout = 0.3)

# resolution = 300x300
def EfficientNetB3():
	return EfficientNet(width_gain = 1.2, depth_gain = 1.4, dropout = 0.3)

# resolution = 380x380
def EfficientNetB4():
	return EfficientNet(width_gain = 1.4, depth_gain = 1.8, dropout = 0.4)

# resolution = 456x456
def EfficientNetB5():
	return EfficientNet(width_gain = 1.6, depth_gain = 2.2, dropout = 0.4)

# resolution = 528x528
def EfficientNetB6():
	return EfficientNet(width_gain = 1.8, depth_gain = 2.6, dropout = 0.5)

# resolution = 600x600
def EfficientNetB7():
	return EfficientNet(width_gain = 2.0, depth_gain = 3.1, dropout = 0.5)

# resolution = 800x800
def EfficientNetL2():
	return EfficientNet(width_gain = 4.3, depth_gain = 5.3, dropout = 0.5)