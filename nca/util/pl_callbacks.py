import os
from typing import Optional

import imageio
import numpy as np
import pytorch_lightning as pl
import torch

from nca.nn.basenca import BaseSystemNCA
from nca.util.im import im_show, im_row
from nca.util.im import im_to_numpy


# ========================================================================= #
# PyTorch Lightning Callbacks                                               #
# ========================================================================= #


class _PeriodCallback(pl.Callback):

    def __init__(self, period=500):
        self._period = period
        self._count = 0

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self._save(_get_system_nca(pl_module), idx=None)

    def on_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self._period is None:
            return
        self._count += 1
        if self._count % self._period != 0:
            return
        self._save(_get_system_nca(pl_module), idx=self._count)

    def _save(self, nca_system: BaseSystemNCA, idx: Optional[int]):
        raise NotImplementedError


class CallbackImshowNCA(_PeriodCallback):

    def __init__(self, steps=(128, 256, 512, 1024, 2048), period: Optional[int] = 500, figwidth: float = 10, figpadpx: int = 8, forward_kwargs: Optional[dict] = None, start_batch_kwargs: Optional[dict] = None):
        super().__init__(period=period)
        self._figwidth = figwidth
        self._figpadpx = figpadpx
        self._steps = {steps} if isinstance(steps, int) else set(steps)
        # function kwargs
        self._forward_kwargs = forward_kwargs
        self._start_batch_kwargs = start_batch_kwargs

    def _save(self, nca_system: BaseSystemNCA, idx: Optional[int]):
        # visualise -- TODO: add wandb support
        images = _yield_nca_frames(nca_system, frame_numbers=self._steps, start_batch_kwargs=self._start_batch_kwargs, forward_kwargs=self._forward_kwargs)
        im_show(im_row(images, pad=self._figpadpx), figwidth=self._figwidth, title='The End' if idx is None else f'Step: {idx}')


class CallbackVidsaveNCA(_PeriodCallback):

    def __init__(self, save_dir: str, period: Optional[int] = None, forward_kwargs: Optional[dict] = None, start_batch_kwargs: Optional[dict] = None):
        super().__init__(period=period)
        self._save_dir = save_dir
        # function kwargs
        self._forward_kwargs = forward_kwargs
        self._start_batch_kwargs = start_batch_kwargs

    def _save(self, nca_system: BaseSystemNCA, idx: Optional[int]) -> None:
        # prepare the file
        name = 'visualise.mp4' if (idx is None) else f'visualise_{idx}.mp4'
        path = os.path.join(self._save_dir, name)
        os.makedirs(self._save_dir, exist_ok=True)

        # visualise -- TODO: add wandb support
        frames = _yield_fast_forward_nca_frames(nca_system, start_batch_kwargs=self._start_batch_kwargs, forward_kwargs=self._forward_kwargs)
        with imageio.get_writer(path, fps=30) as writer:
            for frame in frames:
                writer.append_data(frame)


# ========================================================================= #
# Frame Generators                                                          #
# ========================================================================= #


def _get_system_nca(pl_module: 'pl.LightningModule') -> BaseSystemNCA:
    if not isinstance(pl_module, BaseSystemNCA):
        raise TypeError(f'pl_module is not an instance of: {BaseSystemNCA.__name__}, got: {type(pl_module)}')
    return pl_module


def _yield_fast_forward_nca_frames(pl_module, dtype=np.uint8, start_batch_kwargs=None, forward_kwargs=None):
    # 3330 total iterations
    frame_numbers = np.cumsum(np.minimum(2**(np.arange(300) // 30), 16))
    yield from _yield_nca_frames(pl_module, frame_numbers=frame_numbers, dtype=dtype, start_batch_kwargs=start_batch_kwargs, forward_kwargs=forward_kwargs)


def _yield_nca_frames(pl_module, frame_numbers, dtype=np.uint8, start_batch_kwargs=None, forward_kwargs=None):
    # get default arguments
    if start_batch_kwargs is None: start_batch_kwargs = {}
    if forward_kwargs is None: forward_kwargs = {}
    # get frame numbers
    frame_numbers = set(int(i) for i in frame_numbers if int(i) >= 0)
    assert frame_numbers, 'frame numbers cannot be empty'
    # get system
    nca_system = _get_system_nca(pl_module)
    # generate images
    with torch.no_grad():
        x = nca_system.make_start_batch(batch_size=1, **start_batch_kwargs)
        for i in range(max(frame_numbers) + 1):
            x, frame = nca_system.forward(x, iterations=1, return_img=True, **forward_kwargs)
            if i in frame_numbers:
                yield im_to_numpy(frame[0], dtype=dtype)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
