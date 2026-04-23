import numpy as np

import torch.nn as nn


def graph_breaking_mask(x, lengths):
    N = x.size(0)
    mask = x.new_ones(x.size()[:-1])
    for i in range(N):
        mask[i, lengths[i] :] = 0
    return mask.unsqueeze(-1)


def attn_pad_mask(x, lengths, expand_length):
    mask = graph_breaking_mask(x, lengths=lengths)
    pad_mask = mask.squeeze(-1).lt(1)
    return pad_mask.unsqueeze(1).expand(-1, expand_length, -1)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(64, 64)
        self.scale = np.power(64, 0.5)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        return self.norm(self.w(x) / self.scale + x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 64)
        self.layer_stack = nn.ModuleList([Layer()])

    def forward(self, x, lengths):
        mask = graph_breaking_mask(x, lengths=lengths)
        length = x.size(1)
        attn_mask = attn_pad_mask(x, lengths, length)  # noqa: F841
        out = self.linear(x)
        for layer in self.layer_stack:
            out = layer(out) * mask
        return (out,)
