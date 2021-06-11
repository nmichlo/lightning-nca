import torch
from torch import nn
from torch.nn import functional as F


# ========================================================================= #
# Channel-Wise Filters                                                      #
# ========================================================================= #


class ChannelConv(nn.Module):

    def __init__(self, filters, requires_grad=False):
        super().__init__()
        # not-learnable
        self._filters = nn.Parameter(torch.stack(filters), requires_grad=requires_grad)
        N, H, W = self._filters.shape
        assert H == W == 3

    def forward(self, x):
        b, ch, h, w = x.shape
        y = x.reshape(b * ch, 1, h, w)
        y = F.pad(y, [1, 1, 1, 1], 'circular')
        y = F.conv2d(y, self._filters[:, None])
        return y.reshape(b, -1, h, w)  # (B, C, H, W) -> (B, C*FILTER_N, H, W)


class PresetFilters(ChannelConv):
    """
    pre-defined filters from:
    https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_pytorch.ipynb
    """

    def __init__(self, requires_grad=False):
        identity = torch.tensor([[ 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0]])
        laplace  = torch.tensor([[ 1.0, 2.0, 1.0], [ 2.0, -12, 2.0], [ 1.0, 2.0, 1.0]]) / 16.0
        sobel_x  = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y  = sobel_x.T
        super().__init__([identity, sobel_x, sobel_y, laplace], requires_grad=requires_grad)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
