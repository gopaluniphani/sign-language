import torch
import torch.nn as nn
import torchvision.models as models

from TemporalConvNet import TemporalBlock

class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        self.tcl = TemporalBlock()
        self.blstm = nn.LSTM(
            input_size=1000,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.network = nn.Sequential(
            self.inception,
            self.tcs,
            self.blstm
        )

    def forward():
        return self.network()
    