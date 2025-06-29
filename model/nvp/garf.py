import torch
from torch import nn
import numpy as np


class GaussianLayer(nn.Module):
    def __init__(self, in_dims, out_dims, bias=True, sigma=0.1):
        super().__init__()
        self.in_dims = in_dims
        self.linear = nn.Linear(in_dims, out_dims, bias=bias)
        self.sigma = sigma

    def forward(self, x):
        x = self.linear(x)
        return torch.exp(-x**2/(2* self.sigma**2))


class Garf(nn.Module):
    def __init__(self, in_dims, hidden_dims, hidden_layers, out_dims, sigma=0.1):
        super().__init__()

        self.net = []
        self.net.append(GaussianLayer(in_dims, hidden_dims, sigma))

        for i in range(hidden_layers):
            self.net.append(GaussianLayer(hidden_dims, hidden_dims, sigma))

        final_linear = nn.Linear(hidden_dims, out_dims)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)