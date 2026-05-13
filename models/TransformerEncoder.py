import torch
from torch import nn
from models.MultiHeadAttention import MultiHeadAttention
from models.LayerNorm import LayerNorm
from models.FeedForwordLayer import FeedForwardLayer

# 首先定义一个TransformerBlock模块，Encoder只是将其重复num_encoder_layers次
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head, feature_size, slide_win, forward_expansion, device, dropout, max_length=1024):
        super(TransformerBlock, self).__init__()
        self.device = device
        self.attn = MultiHeadAttention(embed_size, head)
        self.norm1 = LayerNorm(feature_size)
        self.norm2 = LayerNorm(feature_size)
        self.feed_forward = FeedForwardLayer(feature_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, mask1=None):
        attention, score = self.attn(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            feature_size,
            slide_win,
            forward_expansion,
            device = 'cpu',
            dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, feature_size, slide_win, forward_expansion, device, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, mask1=None):
        for layer in self.layers:
            x = layer(x, x, x, mask, mask1)
        return x


