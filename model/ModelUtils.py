from functools import partial

import torch
from torch import nn
from einops import reduce
import torch.nn.functional as F


# conv layer with weight standardisation incorporated
class WeightStandardizedConv2d(nn.Conv2d):
    # no need to override init, we will just override forward

    def forward(self, x):
        # eps is added to denominator to prevent division by zero
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=True))

        weight = (weight - mean) / (var + eps).rsqrt()

        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

def exists(val):
    return val is not None
