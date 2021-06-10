# pytorch lightning port of:
# https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_pytorch.ipynb

from typing import Iterator

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from torchvision.models import vgg16
from torchvision.transforms import Normalize


# ========================================================================= #
# Channel-Wise Filters                                                      #
# ========================================================================= #
from nca.common import im_read
from nca.common import VisualiseNCA


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


class DefaultFilters(ChannelConv):
    def __init__(self, requires_grad=False):
        identity = torch.tensor([[ 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0]])
        laplace  = torch.tensor([[ 1.0, 2.0, 1.0], [ 2.0, -12, 2.0], [ 1.0, 2.0, 1.0]]) / 16.0
        sobel_x  = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y  = sobel_x.T
        super().__init__([identity, sobel_x, sobel_y, laplace], requires_grad=requires_grad)


# ========================================================================= #
# END                                                                       #
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
        xs = self._compute_styles(x)
        # compute loss
        loss = 0.0
        for x, y in zip(xs, self._style_targets):
            loss += (x - y).square().mean()
        return loss

    def _compute_styles(self, imgs):
        x = self._normalise(imgs)
        grams = []
        for i, layer in enumerate(self._feature_layers):
            x = layer(x)
            if i in self._layer_indices:
                h, w = x.shape[-2:]
                y = x.clone()  # workaround for pytorch in-place modification bug(?)
                gram = torch.einsum('bchw, bdhw -> bcd', y, y) / (h * w)
                grams.append(gram)
        return grams


# ========================================================================= #
# Neural Cellular Automata                                                  #
# ========================================================================= #


class NCA(nn.Module):

    def __init__(self, channels=12, hidden_channels=96, learn_filters=False):
        super().__init__()
        # not-learnable
        self.chn = channels
        # learnable layers
        if learn_filters:
            self.filter = ChannelConv(list(torch.randn(4, 3, 3)), requires_grad=True)  # not in original
        else:
            self.filter = DefaultFilters(requires_grad=False)
        # learnable secondary layers
        self.w1 = torch.nn.Conv2d(in_channels=channels * 4, out_channels=hidden_channels, kernel_size=1)
        self.w2 = torch.nn.Conv2d(in_channels=hidden_channels, out_channels=channels, kernel_size=1, bias=False)
        self.w2.weight.data.zero_()

    def forward(self, x, iterations=1, update_ratio=0.5, return_img=False):
        for _ in range(iterations):
            x = self._forward_single(x, update_ratio=update_ratio)
        if return_img:
            return x, self.extract_rgb(x)
        return x

    def _forward_single(self, x, update_ratio=0.5):
        # generate random update mask
        B, C, H, W = x.shape
        update_mask = torch.rand(B, 1, H, W, device=x.device) < update_ratio
        # feed forward
        y = self.w2(torch.relu(self.w1(self.filter(x))))
        return x + y * update_mask

    def make_start_organisms(self, batch_size, size=128, device=None):
        return torch.zeros(batch_size, self.chn, size, size, device=device, requires_grad=False)

    @staticmethod
    def extract_rgb(x):
        # why + 0.5?
        return x[..., :3, :, :] + 0.5


# ========================================================================= #
# NCA System                                                                #
# ========================================================================= #


class NcaSystem(pl.LightningModule):

    def __init__(
        self,
        style_img_url='https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Tempera%2C_charcoal_and_gouache_mountain_painting_by_Nicholas_Roerich.jpg/301px-Tempera%2C_charcoal_and_gouache_mountain_painting_by_Nicholas_Roerich.jpg',
        img_size=128,
        # nca options
        nca_learn_filters=False,
        nca_channels=12,
        nca_hidden_channels=96,
        # training options
        iters=(32, 64),    # original is (64, 96)
        batch_size=4,      # original is 4
        lr=1e-3,
        pool_size=1024,
        pool_reset_element_period=2,
        pool_on_cpu=True,  # save GPU memory, especially if the pool is large
        # extra, not configurable in original implementation
        normalize_gradient=True,  # try this off
        consistency_loss=False,   # try this on
        scale_loss=None,          # max loss value if you have stability problems
    ):
        super().__init__()
        self.save_hyperparameters()
        # check hparams
        self.hparams.iters_min, self.hparams.iters_max = (self.hparams.iters, self.hparams.iters) if isinstance(self.hparams.iters, int) else self.hparams.iters
        # create the model
        self.nca = NCA(channels=self.hparams.nca_channels, hidden_channels=self.hparams.nca_hidden_channels, learn_filters=self.hparams.nca_learn_filters)
        # training attributes
        self.style_img = im_read(self.hparams.style_img_url, size=self.hparams.img_size)
        self._style_loss = None
        self._organism_pool = None

    def forward(self, *args, **kwargs):
        return self.nca(*args, **kwargs)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Training Setup                                                        #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def on_train_start(self):
        # make loss & organism pool
        loss = StyleLoss(target_img=self.style_img)
        pool = self.nca.make_start_organisms(batch_size=self.hparams.pool_size, size=self.hparams.img_size)
        # load to correct device, store pool on CPU to save some memory
        # style loss is too slow on CPU, but uses a fair bit of RAM
        self._style_loss = loss.to(self.device)
        self._organism_pool = pool.to('cpu' if self.hparams.pool_on_cpu else self.device)

    def on_train_end(self):
        self._style_loss = None
        self._organism_pool = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Training                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def training_step(self, idxs, i):
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # load organisms from the pool
        with torch.no_grad():
            # reset runs in the pool
            if i % self.hparams.pool_reset_element_period == 0:
                self._organism_pool[idxs[0]] = self.nca.make_start_organisms(batch_size=1, size=self.hparams.img_size, device=self._organism_pool.device)[0]
            # load incomplete runs from the pool
            x = self._organism_pool[idxs].to(self.device)
            x_img = self.nca.extract_rgb(x)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # run cellular automata
        y, y_img = self.nca.forward(x, return_img=True, iterations=np.random.randint(self.hparams.iters_min, self.hparams.iters_max + 1))
        # compute cellular loss
        loss = self._style_loss(y_img)

        # organisms should minimize drift
        if self.hparams.consistency_loss:
            loss += F.mse_loss(y_img, x_img)

        # make sure the loss is not too large, especially prominent if we don't normalise the gradient
        if self.hparams.scale_loss is not None:
            if loss > self.hparams.scale_loss:
                print(f'warning: scaling loss: {loss.item()} -> {self.hparams.scale_loss}')
                with torch.no_grad():
                    scale = self.hparams.scale_loss / loss
                loss *= scale

        # normalize gradients
        if self.hparams.normalize_gradient:
            loss.backward(retain_graph=True)
            for p in self.nca.parameters():
                if (p is not None) and (p.grad is not None):
                    p.grad /= (p.grad.norm() + 1e-8)  # normalize gradients

        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # save the incomplete runs
        with torch.no_grad():
            self._organism_pool[idxs] = y.to('cpu' if self.hparams.pool_on_cpu else self.device)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # done!
        return loss

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Helper                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": MultiStepLR(optimizer, [2000], 0.3),
            "monitor": "recon",
        }

    def train_dataloader(self):
        # returns random indices of the pool
        class IndexDataset(IterableDataset):
            def __iter__(this) -> Iterator[T_co]:
                while True:
                    yield np.random.choice(self.hparams.pool_size, self.hparams.batch_size, replace=False)
        # we modify the collect function so that the batch indices are of shape (,B) rather than (1, B)
        # we could set the batch size in the dataloader, but then the indices might not be unique
        # we could also use RandomSampler instead, with a dataset that just returns the index from __get_item__, but we don't have a concept of epochs.
        return DataLoader(IndexDataset(), num_workers=0, batch_size=1, collate_fn=lambda x: torch.as_tensor(x)[0])


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':

    trainer = pl.Trainer(
        max_steps=10000,
        checkpoint_callback=False,
        logger=False,
        callbacks=[VisualiseNCA(period=250, img_size=256)],
        gpus=1,
    )

    # default settings use about 5614MiB of RAM
    system = NcaSystem(
        style_img_url='yarn_ball.png',
        # extras
        consistency_loss=True,    # not enabled in original version
        normalize_gradient=False, # enabled in original version
    )

    trainer.fit(system)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
