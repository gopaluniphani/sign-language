import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:,:,: - self.chomp_size].contiguous()
        
class TemporalBlock(nn.Module):
    def __init__(self):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(1, 1, 5,
                                           stride=1, padding=4, dilation=1))
        self.chomp1 = Chomp1d(4)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = weight_norm(nn.Conv1d(1, 1, 5,
                                           stride=1, padding=4, dilation=2))
        self.chomp2 = Chomp1d(4)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)