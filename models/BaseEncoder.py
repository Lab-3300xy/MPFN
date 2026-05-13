import torch
from torch import nn
from models.MultiHeadAttention import MultiHeadAttention
from models.LayerNorm import LayerNorm
from models.FeedForwordLayer import FeedForwardLayer
# from torch_geometric.nn import GATConv
# from models.trans_gat.models_type import get_edge

from models.FourierBlock import FourierBlock

# 首先定义一个TransformerBlock模块，Encoder只是将其重复num_encoder_layers次
class TransformerBlock(nn.Module):
    def __init__(self, pred_len, embed_size, head, feature_size, slide_win, forward_expansion, device, dropout, max_length=1024):
        super(TransformerBlock, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.attn_t = MultiHeadAttention(embed_size, head)
        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)
        self.feed_forward = FeedForwardLayer(embed_size, forward_expansion)

        self.dropout = nn.Dropout(dropout)

        self.fft = FourierBlock(feature_size, embed_size)

        self.attn_c = MultiHeadAttention(embed_size, head)

        self.map = nn.Linear(feature_size, embed_size)
        self.reserve_map = nn.Linear(embed_size, feature_size)



    def forward(self, query, key, value, mask=None, mask1=None):

        x = query
        N, L, H = x.shape
        attention_spatial = self.attn_c(query, key, value, mask1)

        x2 = x + attention_spatial
        x2 = x2.transpose(1,2)

        fft_out = self.fft(x2)

        x2 = self.dropout(
            self.map(x2 + fft_out)
        )
        query, key, value = x2, x2, x2
        attention = self.attn_t(query, key, value, mask)

        x3 = x2 + attention
        x3 = self.reserve_map(x3)
        x4 = self.dropout(self.norm1(x3.transpose(1, 2)))
        forward = self.feed_forward(x4)
        out = self.dropout(self.norm2(forward + x4))

        return out

class Encoder(nn.Module):
    def __init__(
            self,
            pred_len,
            embed_size,
            num_layers,
            heads,
            feature_size,
            slide_win,
            forward_expansion,
            device = 'cuda',
            dropout=0.1):
        super(Encoder, self).__init__()

        self.layer = TransformerBlock(pred_len, embed_size, heads, feature_size, slide_win, forward_expansion, device, dropout)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None, mask1=None):
        x = self.layer(x, x, x, mask, mask1)
        x = self.norm(x) #zhiqianmeiyou
        return x


