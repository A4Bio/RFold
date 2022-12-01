import torch
from torch import nn, einsum
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,residual=False):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.residual = residual

    def forward(self,x):
        if self.residual:
            return x + self.conv(x)
        return self.conv(x)
        

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)


class Attn(nn.Module):
    def __init__(self, *, dim, query_key_dim=128, expansion_factor=2.,
        dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )
        self.offsetscale = OffsetScale(query_key_dim, heads=2)

    def forward(self, x):
        seq_len = x.shape[-2]
        normed_x = self.norm(x)
        qk = self.to_qk(normed_x)
        q, k = self.offsetscale(qk)
        sim = einsum('b i d, b j d -> b i j', q, k) / seq_len
        attn = F.relu(sim) ** 2
        return attn