import math
import torch
from torch import nn
from models.MultiHeadAttention import MultiHeadAttention
from models.Transpose import *




class DBLIEM(nn.Module):
    def __init__(self, embed_size, feature_size, num_cross_layer, patch_kernel_size, patch_stride, time_seq, conv_kernel_size,
                 conv_stirde, num_intra_layer, device='cuda',):
        super(DBLIEM, self).__init__()
        self.patch_kernel_size = patch_kernel_size
        self.embed_size = embed_size
        self.time_seq = time_seq
        self.patch_stride = patch_stride
        self.device = device
        self.patch_stride = patch_stride

        self.conv_padding = conv_kernel_size // 2

        self.patch_layer = nn.Unfold(kernel_size=patch_kernel_size, stride=patch_stride, padding=0, dilation=1)
        self.rev_patch_layer = nn.Fold(output_size=(time_seq, embed_size),kernel_size=patch_kernel_size, stride=patch_stride, padding=1)

        self.cross_branch = nn.ModuleList([
            MultiHeadAttention(patch_kernel_size * patch_kernel_size, 1)
            for _ in range(num_cross_layer)
        ])

        L1 = math.floor((embed_size - patch_kernel_size + 2) / patch_stride + 1)
        L2 = math.floor((time_seq - patch_kernel_size +2) / patch_stride + 1)
        L = L1 * L2

        self.dropout = nn.Dropout(0.1)

        self.intra_branch = nn.ModuleList([
            nn.Conv1d(L, L, kernel_size=conv_kernel_size, stride=conv_stirde, padding=0, dilation=1)
            for _ in range(num_intra_layer)
        ])


        # self.src_position_embedding = nn.Embedding(1024, patch_kernel_size * patch_kernel_size)
        self.src_position_embedding = nn.Parameter(torch.zeros(L,  patch_kernel_size * patch_kernel_size))

        self.linear = nn.Linear(feature_size, embed_size)
        self.rec_linear = nn.Linear(embed_size, feature_size)

        self.norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(feature_size), Transpose(1, 2))
        self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(feature_size), Transpose(1, 2))
        self.num_gol = 128
        self.rev_conv_map = nn.Conv1d(self.num_gol, L, kernel_size=conv_kernel_size, stride=conv_stirde, padding=self.conv_padding, dilation=1)
        self.query = nn.Parameter(torch.randn(1, self.num_gol, patch_kernel_size * patch_kernel_size).to('cuda'))

        nn.init.xavier_normal_(self.src_position_embedding)

    def forward(self, input):
        # print(input.shape)
        _, c, _ = input.shape
        input = self.norm(input)
        input = self.linear(input)
        input = input.unsqueeze(1)
        input = nn.ReflectionPad2d(padding=(1, 1, 1, 1))(input)
        input = self.patch_layer(input)
        x = input.transpose(2,1)
        b, h, w = x.shape
        x2 = x
        for layer in self.intra_branch:
            x2 = nn.ReplicationPad1d(padding=(self.conv_padding, self.conv_padding))(x2)
            x2 = layer(x2)
        x1 = x.reshape(-1, h, w)

        # src_positions = (
        #     torch.arange(0, h)
        #     .unsqueeze(0)  # 在张量最前面加一个维度
        #     .expand(b, h)  #
        #     .to(self.device)
        # )

        x1 = self.dropout(
            x1 + self.src_position_embedding
            # x1 + self.src_position_embedding(src_positions)
        )

        query = self.query.repeat(b, 1, 1)
        for MHA in self.cross_branch:
            x1 = MHA(query, x1, x1)

        x1 = self.rev_conv_map(x1)
        out = self.rev_patch_layer((x1.reshape(-1, h, w) + x2.reshape(-1, h, w)).transpose(2, 1)) #11111111111111111111111111111
        out = out.reshape(b, c, -1)
        out = self.rec_linear(out)
        out = self.norm1(out)
        return out
