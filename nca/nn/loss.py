import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.transforms import Normalize


# ========================================================================= #
# Stylistic Loss                                                            #
# ========================================================================= #


class StyleLoss(nn.Module):

    def __init__(self, target_img, style_layers=(1, 6, 11, 18, 25)):
        super().__init__()
        assert target_img.ndim == 3
        assert target_img.dtype == torch.float32
        # layers to use for computing the style
        self._layer_indices = style_layers
        # not-learnable:
        with torch.no_grad():
            self._normalise      = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            self._feature_layers = nn.ModuleList(vgg16(pretrained=True).features[:max(self._layer_indices)+1])
            self._style_targets  = nn.ParameterList(nn.Parameter(t) for t in self._compute_styles(target_img[None]))

    def forward(self, x):
        # compute loss as the sum of the means
        # of the different layer outputs of the vgg net
        loss = 0.0
        for x, y in zip(self._compute_styles(x), self._style_targets):
            loss += F.mse_loss(x, y, reduction='mean')
        return loss

    def _compute_styles(self, imgs):
        x = self._normalise(imgs)
        grams = []
        for i, layer in enumerate(self._feature_layers):
            x = layer(x)
            if i in self._layer_indices:
                h, w = x.shape[-2:]
                y = x.clone()  # workaround in-place bug?
                gram = torch.einsum('bchw, bdhw -> bcd', y, y) / (h * w)
                grams.append(gram)
        return grams


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
