import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=512,
            batch_first=True
        )
        self.network = nn.Sequential(
            self.embedding,
            self.lstm
        )

    def forward(self, x):
        return self.network(x)