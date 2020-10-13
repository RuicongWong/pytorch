import torch
from torch import nn
from torch.nn import functional as F
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=ch_out)
        self.conv2 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=ch_out)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return F.relu(out)
class ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        for i in range(2):
            self.conv.add_module(name='ResBlk' + chr(i), module=ResBlk(ch_in=64, ch_out=64, stride=1))
        self.conv.add_module(name='ResBlk2', module=ResBlk(ch_in=64, ch_out=128, stride=2))
        self.conv.add_module(name='ResBlk3', module=ResBlk(ch_in=128, ch_out=128, stride=1))
        self.conv.add_module(name='ResBlk4', module=ResBlk(ch_in=128, ch_out=256, stride=2))
        self.conv.add_module(name='ResBlk5', module=ResBlk(ch_in=256, ch_out=256, stride=1))
        self.conv.add_module(name='ResBlk6', module=ResBlk(ch_in=256, ch_out=512, stride=2))
        self.conv.add_module(name='ResBlk7', module=ResBlk(ch_in=512, ch_out=512, stride=1))
        self.outlayer = nn.Linear(in_features=512, out_features=num_classes)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x



