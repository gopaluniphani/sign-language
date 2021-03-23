import torch
import torch.nn as nn

import VideoEncoder
import TextEncoder

class Model(nn.Module):
    def __init__(self):
        self.videoEncoder = VideoEncoder()
        self.textEncoder = TextEncoder()

    def forward(self, x):
        return [self.videoEncoder(x[0]), self.textEncoder(x[1])]