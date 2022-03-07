import torch
import torch.nn as nn

NUM_FILTERS = 5
FILTER_SIZE = 3
DELTA = 0.5

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, delta=0.1):
        super(BasicBlock, self).__init__()

        self.delta = delta
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=FILTER_SIZE,padding=1)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = torch.tanh(out)
        out = self.delta*out + identity
        return out


class SmallResNet(nn.Module):

    def __init__(self, input_dim=1, num_blocks=2, num_classes=10):
        super(SmallResNet, self).__init__()

        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.blocks = []

        ###
        self.conv1 = nn.Conv2d(in_channels=input_dim,out_channels=NUM_FILTERS,kernel_size=FILTER_SIZE,padding=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2 = nn.Conv2d(in_channels=NUM_FILTERS,out_channels=NUM_FILTERS,kernel_size=FILTER_SIZE,padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        self.fc = nn.Linear(49,self.num_classes)

        for i in range(self.num_blocks):
            block = BasicBlock(in_channels=NUM_FILTERS, out_channels=NUM_FILTERS, delta=DELTA)
            self.blocks.append(block)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.tanh(out)
        out = self.avgpool1(out)
        out = self.conv2(out)
        out = torch.tanh(out)
        out = self.avgpool2(out)

        for i in range(self.num_blocks):
            block = self.blocks[i]
            out = block(out)

        out = torch.mean(out,dim=1)
        out = torch.flatten(out,start_dim=1)
        out = self.fc(out)
        return out


def smallresnet(**kwargs):
    return SmallResNet(**kwargs)

