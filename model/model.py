# This is just a u-net

import torch
from torch import nn
from model.ModelUtils import WeightStandardizedConv2d, exists

# Basic block with conv + groupNorm + Silu act
class Block(nn.Module):
    def __init__(self, dim_in, dim_out, groups_count=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim_in, dim_out, 3, 1)
        self.norm = nn.GroupNorm(groups_count, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale+1) + shift

        x = self.act(x)
        return x