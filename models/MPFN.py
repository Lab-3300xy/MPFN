from models.BaseEncoder import Encoder
from models.DBLIEM import DBLIEM
from models.InsNorm import InsNorm
import numpy as np


from models.Transpose import *
torch.set_printoptions(threshold=np.inf)

class MPFN(nn.Module):
    def __init__(
            self,
            embed_size,
            slide_win,
            pred_len,
            feature_size,
            num_encoder_layers=2,
            num_cross_layers = 1,
            num_intra_layers = 1,
            forward_expansion=4,
            patch_kernel_size=7,
            patch_stride=5,
            conv_kernel_size=3,
            heads=8,
            dropout=0.1,
            dropout1=0.7,
            device='cuda',
            dataName = '',
            src_pad_idx=None,
            ):
        super(MPFN, self).__init__()

        self.src_pad_idx = None
        if src_pad_idx is not None:
            self.src_pad_idx = src_pad_idx
        self.num_encoder_layers = num_encoder_layers
        self.feature_size = feature_size
        self.pred_len=pred_len
        self.device = device
        self.dataName = dataName


        self.FTCT = nn.ModuleList([
            Encoder(pred_len=pred_len, embed_size=embed_size, num_layers=1, heads=heads, feature_size = feature_size,
                    slide_win = slide_win, forward_expansion=forward_expansion, device=device, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.channel_embedding = nn.Parameter(torch.zeros(feature_size, embed_size))
        self.linear = nn.Linear(embed_size, slide_win)

        self.map = nn.Linear(slide_win, embed_size)

        self.out_layer = nn.Linear(embed_size, pred_len)
        self.out_layer1 = nn.Linear(slide_win, pred_len)


        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout1)

        self.DBLIEM = DBLIEM(embed_size=embed_size, feature_size=feature_size, num_cross_layer=num_cross_layers,
                             patch_kernel_size=patch_kernel_size, patch_stride=patch_stride, time_seq=slide_win,
                             conv_kernel_size=conv_kernel_size, conv_stirde=1, num_intra_layer=num_intra_layers)

        self.insNorm = InsNorm(feature_size)

        nn.init.xavier_normal_(self.channel_embedding)


    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        return src_mask.to(self.device)


    def forward(self, x, mode='train'):

        x = self.insNorm(x, 'norm')
        orig_x = x


        src_mask = None
        if self.src_pad_idx is not None:
            src_mask = self.make_src_mask(x)

        x = self.map(x.transpose(1, 2))

        x = self.dropout(
            x + self.channel_embedding
        )
        out = torch.zeros(x.shape).to(self.device)
        for layer in self.FTCT:
            if src_mask is not None:
                encoder_out = layer(x, src_mask)
                out = out + encoder_out
                x = x - encoder_out
            else:
                encoder_out = layer(x, src_mask)
                out = out + encoder_out
                x = x - encoder_out

        out1 = self.DBLIEM(self.linear(out).transpose(1,2) + orig_x)

        out = out + x
        out = self.out_layer(out)
        out = out.permute(0, 2, 1)
        out = self.dropout(out)
        out1 = self.out_layer1(out1.transpose(-2, -1))
        out1 = self.dropout1(out1)
        out1 = out1.view(-1, self.pred_len, self.feature_size)
        out = out + out1
        out = self.insNorm(out, 'denorm')

        return out
