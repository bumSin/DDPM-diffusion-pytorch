# This is just a u-net

import torch
from einops import rearrange
from torch import nn
from model.ModelUtils import WeightStandardizedConv2d, exists

# Most basic block with conv + groupNorm + Silu activation
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

# The residual block
class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim, groups=8):
        super().__init__()

        self.block1 = Block(dim_in, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)

        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_out != dim_in else nn.Identity()

        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2) if exists(time_emb_dim)
            else None
        )

    def forward(self, x, time_emb=None):
        scale_shift = None

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)  # break along channel dim in half, from dim_out * 2 to dim_out

        h = self.block1(x, scale_shift)
        h = self.block2(h)
        h = h + self.res_conv(x)

        return h