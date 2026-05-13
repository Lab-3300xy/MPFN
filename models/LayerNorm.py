import torch
from torch import nn



class LayerNorm(nn.Module):

    def __init__(self,  feature_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(feature_size))
        self.b = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a * (x- mean) / (std + self.eps) + self.b
