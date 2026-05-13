import torch
import torch.nn as nn
import pandas as pd

class InsNorm(nn.Module):
    def __init__(self, feature_size, eps = 1e-5):
        super(InsNorm, self).__init__()
        self.feature_size = feature_size
        self.eps = eps
        self.a = nn.Parameter(torch.ones(feature_size))
        self.b = nn.Parameter(torch.zeros(feature_size))
        self.mean = None
        self.stdev = None


    def forward(self, x, mode):
        if mode == 'norm':
            self.get_statistics(x)
            x = self.normlize(x)
        elif mode == 'denorm':
            x = self.denormlize(x)
        else: raise NotImplemented
        return x

    def get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        # dim2reduce = 2
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()


    def normlize(self, x):
        x = x - self.mean
        x = x / self.stdev
        x = x * self.a
        x = x + self.b
        return x

    def denormlize(self, x):
        x = x - self.b
        x = x / (self.a + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
