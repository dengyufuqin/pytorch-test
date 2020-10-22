import torch
import torch.nn as nn
from utils.utils import *
from torch.nn import init

# SE-ResNet
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, bias=False)


# SE convolution
class SELayer(nn.Module):
    def __init__(self, in_channels, reduce=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduce, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduce, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.avgpool(x)
        out = self.fc(out)
        return x * out


class ResidualBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels*self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.downsample =downsample
        self.se = SELayer(out_channels*self.expansion)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1)
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, block.expansion * out_channels, stride=stride),
                nn.BatchNorm2d(block.expansion * out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



# MobileNetV1

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        def conv_bn(in_channel, out_channel, stride):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(in_channel, out_channel, stride):
            return nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 3, stride, padding=1, groups=in_channel, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU6(inplace=True),

                nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU6(inplace=True)
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x


# MobileNetV2
class Head(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Head, self).__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu6(x)

        return x


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expansion):
        super(BottleNeck, self).__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel
        channels = expansion * in_channel
        self.conv1 = nn.Conv2d(in_channel, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu6 = nn.ReLU6(inplace=True)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu6(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu6(x)
        x = self.conv3(x)
        x = self.bn3(x)

        out = x + self.shortcut(residual) if self.stride == 1 else x
        return out
#
class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1, num_classes=2):
        super(MobileNetV2, self).__init__()
        block = BottleNeck
        in_channel = 32
        bottleNeck_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 32, 2],
            [6, 320, 1, 1]
        ]
        in_channel = int(in_channel * width_mult)
        head_layer = Head(3, in_channel)
        self.layers = [head_layer]
        for t, c, n, s in bottleNeck_residual_setting:
            stride = s
            out_channel = int(c*width_mult)
            for i in range(n):
                if i == 0:
                    self.layers.append(block(in_channel, out_channel, stride, t))
                else:
                    self.layers.append(block(in_channel, out_channel, 1, t))
                in_channel = out_channel

        self.layers = nn.Sequential(*self.layers)

        self.conv_end = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_end = nn.BatchNorm2d(1280)
        self.relu = nn.ReLU6(inplace=True)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        x = self.relu(x)
        x = self.AdaptiveAvgPool(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out



# MobileNetV3
class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size//reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size//reduction),
            hSwish(),
            nn.Conv2d(in_size//reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hSwish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hSwish(), None, 2),
            Block(3, 80, 200, 80, hSwish(), None, 1),
            Block(3, 80, 184, 80, hSwish(), None, 1),
            Block(3, 80, 184, 80, hSwish(), None, 1),
            Block(3, 80, 480, 112, hSwish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hSwish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hSwish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hSwish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hSwish(), SeModule(160), 1),

        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hSwish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hSwish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out














