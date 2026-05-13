import torch
import torch.nn as nn

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourierBlock, self).__init__()
        print("fourier blcok used!")

        self.len = in_channels

        self.scale = (1 / (in_channels * out_channels))

        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, dtype=torch.float)
        )

    def forward(self, x):
        x = torch.fft.fft(x, dim=-1)
        # print(x.shape)
        # print(self.weights.shape)
        weights = torch.complex(self.weights, torch.zeros_like(self.weights).to(self.weights.device)).T
        real_part = (x.real * weights.real) - (x.imag * weights.imag)
        imag_part = (x.real * weights.imag) + (x.imag * weights.real)

        fft_out = torch.complex(real_part, imag_part)
        fft_out = torch.fft.irfft(fft_out, n=self.len)
        # print(fft_out)
        return fft_out

# class FourierBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(FourierBlock, self).__init__()
#         print("fourier blcok used!")
#
#         self.len = out_channels
#
#         self.scale = (1 / (in_channels * out_channels))
#
#         self.weights = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, dtype=torch.float)
#         )
#
#     def forward(self, x):
#         x = torch.fft.rfft(x, dim=-1)
#         # print(x.shape)
#         weights = torch.complex(self.weights, torch.zeros_like(self.weights).to(self.weights.device))
#         # print(weights.real.shape)
#         order = "bni, io -> bno"
#         real_part = torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag)
#         imag_part = torch.einsum(order, x.real, weights.imag) - torch.einsum(order, x.imag, weights.real)
#
#         fft_out = torch.complex(real_part, imag_part)
#         fft_out = torch.fft.irfft(fft_out, n=self.len)
#         # print(fft_out.shape)
#         return fft_out