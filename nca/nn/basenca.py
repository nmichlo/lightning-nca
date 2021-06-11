import torch.nn as nn
import pytorch_lightning as pl


# ========================================================================= #
# Base Neural Cellular Automata                                             #
# ========================================================================= #


class BaseNCA(nn.Module):

    def forward(self, x, iterations=1, return_img=False, **kwargs):
        for _ in range(iterations):
            x = self._forward_single_iter(x, **kwargs)
        if return_img:
            return x, self.extract_rgb(x)
        return x

    def _forward_single_iter(self, x, **kwargs):
        raise NotImplementedError()

    def make_start_batch(self, batch_size, device=None, **kwargs):
        raise NotImplementedError()

    @classmethod
    def extract_rgb(cls, x):
        # +0.5 so that outputs are centered around zero from the nn
        return x[..., :3, :, :] + 0.5


class BaseSystemNCA(pl.LightningModule, BaseNCA):

    def _forward_single_iter(self, x, **kwargs):
        raise RuntimeError('this should never be called')

    def forward(self, x, iterations=1, return_img=False, **kwargs):
        return self.nca.forward(x, iterations=iterations, return_img=return_img, **kwargs)

    def make_start_batch(self, batch_size, **kwargs):
        return self.nca.make_start_batch(batch_size=batch_size, device=self.device, **kwargs)

    @property
    def nca(self) -> BaseNCA:
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
