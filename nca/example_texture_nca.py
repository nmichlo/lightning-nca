# pytorch lightning port of:
# https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_pytorch.ipynb

from datetime import datetime
from functools import wraps
from typing import Iterator
from typing import Optional
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

from nca.nn.basenca import BaseSystemNCA
from nca.nn.filter import ChannelConv
from nca.nn.filter import PresetFilters
from nca.nn.loss import StyleLoss
from nca.nn.basenca import BaseNCA

from nca.util.im import im_read
from nca.util.im import im_show
from nca.util.pl_callbacks import CallbackImshowNCA
from nca.util.pl_callbacks import CallbackVidsaveNCA


# ========================================================================= #
# Neural Cellular Automata                                                  #
# ========================================================================= #


class TextureNCA(BaseNCA):

    def __init__(self, channels=12, hidden_channels=96, learn_filters=False, pad_mode='circular', default_update_ratio=0.5):
        super().__init__()
        # not-learnable
        self.chn = channels
        self._default_update_ratio = default_update_ratio
        # learnable layers
        if learn_filters:
            self.filter = ChannelConv(list(torch.randn(4, 3, 3)), pad_mode=pad_mode, requires_grad=True)  # not in original
        else:
            self.filter = PresetFilters(pad_mode=pad_mode, requires_grad=False)
        # learnable secondary layers
        self.w1 = torch.nn.Conv2d(in_channels=channels * 4, out_channels=hidden_channels, kernel_size=1)
        self.w2 = torch.nn.Conv2d(in_channels=hidden_channels, out_channels=channels, kernel_size=1, bias=False)
        self.w2.weight.data.zero_()

    def _forward_single_iter(self, x, update_ratio=None):
        if update_ratio is None:
            update_ratio = self._default_update_ratio
        # generate random update mask
        B, C, H, W = x.shape
        update_mask = torch.rand(B, 1, H, W, device=x.device) < update_ratio
        # feed forward
        y = self.w2(torch.relu(self.w1(self.filter(x))))
        return x + y * update_mask

    def make_start_batch(self, batch_size, device=None, size=128):
        return torch.zeros(batch_size, self.chn, size, size, device=device, requires_grad=False)


# ========================================================================= #
# Neural Cellular Automata - Pytorch Lightning System                       #
# ========================================================================= #


class TextureNcaSystem(BaseSystemNCA):

    def __init__(
        self,
        style_img_uri: str = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Tempera%2C_charcoal_and_gouache_mountain_painting_by_Nicholas_Roerich.jpg/301px-Tempera%2C_charcoal_and_gouache_mountain_painting_by_Nicholas_Roerich.jpg',
        style_img_size: int = 128,
        # nca options
        nca_img_size: int = 128,
        nca_learn_filters: bool = False,
        nca_channels: int = 12,
        nca_hidden_channels: int = 96,
        nca_pad_mode: str = 'circular',
        nca_default_update_ratio: float = 0.5,
        # training options
        iters: Tuple[int, int] = (32, 64),    # original is (64, 96)
        batch_size: int = 4,
        lr: float = 1e-3,
        pool_size: int = 1024,
        pool_reset_element_period: int = 2,
        pool_on_cpu: bool = True,                        # save GPU memory, especially if the pool is large
        # extra, not configurable in original implementation
        normalize_gradient: bool = True,                 # try this off
        consistency_loss_scale: float = None,  # try this on
        scale_loss: float = None,              # max loss value if you have stability problems
    ):
        super().__init__()
        self.save_hyperparameters()
        # check hparams
        self.hparams.iters_min, self.hparams.iters_max = (self.hparams.iters, self.hparams.iters) if isinstance(self.hparams.iters, int) else self.hparams.iters
        # create the model
        self._nca = TextureNCA(channels=self.hparams.nca_channels, hidden_channels=self.hparams.nca_hidden_channels, learn_filters=self.hparams.nca_learn_filters, pad_mode=self.hparams.nca_pad_mode, default_update_ratio=self.hparams.nca_default_update_ratio)
        # training attributes
        self.style_img = im_read(self.hparams.style_img_uri, size=self.hparams.style_img_size)
        self._style_loss = None
        self._organism_pool = None

    @property
    def nca(self) -> BaseNCA:
        return self._nca

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Training Setup                                                        #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def on_train_start(self):
        # make loss & organism pool
        loss = StyleLoss(target_img=self.style_img)
        pool = self.nca.make_start_batch(batch_size=self.hparams.pool_size, size=self.hparams.nca_img_size)
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
                self._organism_pool[idxs[0]] = self.nca.make_start_batch(batch_size=1, size=self.hparams.nca_img_size, device=self._organism_pool.device)[0]
            # load incomplete runs from the pool
            x = self._organism_pool[idxs].to(self.device)
            x_img = self.nca.extract_rgb(x)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # run cellular automata
        y, y_img = self.nca.forward(x, return_img=True, iterations=np.random.randint(self.hparams.iters_min, self.hparams.iters_max + 1))
        # compute cellular loss
        loss = self._style_loss(y_img)
        self.log('style', loss, prog_bar=True)

        # organisms should minimize drift
        if self.hparams.consistency_loss_scale is not None:
            consistency_loss = F.mse_loss(y_img, x_img) * self.hparams.consistency_loss_scale
            self.log('consist', consistency_loss, prog_bar=True)
            loss += consistency_loss

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

    class FireRunner(object):
        def __init__(
            self,
            train_steps: int = 5000,
            train_cuda: bool = torch.cuda.is_available(),
            # visualise
            vis_period_plt: int = 500,
            vis_period_vid: int = 2500,
            vis_im_size: int = 256,
            vis_out_dir: str = f'out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
            plt_show: bool = False,
        ):
            self._show = plt_show
            self._trainer = pl.Trainer(
                max_steps=train_steps,
                checkpoint_callback=False,
                logger=False,
                callbacks=[
                    CallbackImshowNCA(period=vis_period_plt, start_batch_kwargs=dict(size=vis_im_size), save_dir=vis_out_dir, show=self._show),
                    CallbackVidsaveNCA(period=vis_period_vid, start_batch_kwargs=dict(size=vis_im_size), save_dir=vis_out_dir)
                ],
                gpus=1 if train_cuda else 0,
            )

        @wraps(TextureNcaSystem.__init__)
        def run(self, **kwargs):
            # initialise system & train
            system = TextureNcaSystem(**kwargs)
            im_show(system.style_img, title='target_img', show=self._show)
            self._trainer.fit(system)

    # entry point
    # eg. $ example_texture_nca.py --help
    # eg. $ example_texture_nca.py run --help
    # eg. $ example_texture_nca.py --train_steps=500 run --lr=0.0005
    import fire
    fire.Fire(FireRunner, name='runner')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
