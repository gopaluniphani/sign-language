import torch
import torch.nn as nn

import VideoEncoder
import TextEncoder

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size):
        self.videoEncoder = VideoEncoder()
        self.textEncoder = TextEncoder(vocab_size, embed_size)

    def forward(self, x):
        return [self.videoEncoder(x[0]), self.textEncoder(x[1])]