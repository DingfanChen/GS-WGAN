import torch.nn as nn
import torch.nn.functional as F

__all__ = ['lenet']

'''
0   conv1   (6,28,28)
1   relu
2   maxpool
3   conv2
4   relu
5   maxpool,
6   flatten
7   fc1
8   relu
9   fc2 
10  relu
11  fc3
'''

class LeNet(nn.Module):
    def __init__(self, input_size=28, nc=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(nc, 6, 5, padding=2 if input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if num_classes <= 2 else num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def get_layers(self):
        layers = [self.conv1, self.relu, self.maxpool,
                  self.conv2, self.relu, self.maxpool,
                  self.flatten, self.fc1, self.relu,
                  self.fc2, self.relu, self.fc3]
        return layers


def lenet(**kwargs):
    return LeNet(**kwargs)
