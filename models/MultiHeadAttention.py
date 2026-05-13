import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def attention(query, key, value, mask=None, droupout=None):
    # 去query最后一维, 即embedding的维数
    d_k = query.size(-1)
    # 按照注意力公式，将query与key的转置相乘，这里面key是将最后两个维度进行转置，再除以缩放系数得到注意力得分张量scores
    # 如果query是[len, embed], 那么socres是[len, len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask(也是[len, len]) 与 score 每个位置一一比较，如果mask[i][j]为0，则将scores[i][j]改为-1e9
        # 负很大的数，在softmax的相当于没有
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对最后一维进行softmax
    scores = F.softmax(scores, dim=-1)

    if droupout is not None:
        scores = droupout(scores)

    #最后，根据公式将p_attn与value张量相乘获得最终的query注意力表示，同时返回权重
    return torch.matmul(scores, value), scores


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, h, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 判断h是否能被d_model整除，这是因为之后要给每个头分配等量的维度特征
        assert d_model % h == 0
        # 得到每个头获得的向量维度d_k
        self.d_k = d_model // h
        self.h = h

        self.w_key = nn.Linear(d_model, d_model)
        self.w_query = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.atten = None # 返回的attention张量，现在还没有，保存给可视化使用

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query = self.w_query(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_key(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_value(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, self.atten = attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # self.draw_attention_score()

        return self.fc_out(x)

    # def draw_attention_score(self):
    #     x = np.arange(1, 121).reshape(1, 120)
    #     x_r = np.repeat(x, [120], axis=0)
    #
    #     y = np.arange(1, 121).reshape(1, 120).T
    #     y_r = np.repeat(y, [120], axis=1)
    #
    #     z = self.atten[1, :, :120, :120].reshape(120, 120)
    #
    #     z = z.detach().cpu().numpy()
    #
    #     start_color = (1, 1, 1.0)
    #     end_color = (0.0, 0.31, 0.63)
    #
    #     # end_color = (0.87, 0.94, 1.0)
    #     # start_color = (0.0, 0.31, 0.63)
    #
    #     custom_cmap = LinearSegmentedColormap.from_list('custom_blue', [start_color, end_color])
    #     c = plt.pcolormesh(x_r, y_r, z, cmap=custom_cmap, shading='gouraud')
    #
    #     plt.colorbar(c, label='AUPR')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.savefig('./attention_score.png')
    #     # plt.show()


