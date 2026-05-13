
from torch import nn
import torch.nn.functional as F


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, forward_expansion):
        super(FeedForwardLayer, self).__init__()
        # self.w1 = nn.Linear(d_model, d_model * forward_expansion)
        # self.w2 = nn.Linear(d_model * forward_expansion, d_model)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model * forward_expansion, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_model * forward_expansion, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(0.0)
        self.activate = F.gelu

    def forward(self, x):
        x = self.dropout(self.activate(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))

        return x
        # return self.w2((F.relu(self.w1(x))))
