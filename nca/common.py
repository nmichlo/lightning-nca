from typing import Union

import imageio
import numpy as np
import PIL
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def im_to_numpy(img: Union[torch.Tensor, np.ndarray], dtype=np.uint8, rgb=True) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.uint8):
        raise ValueError(f'unsupported output dtype: {dtype}, must be {np.uint8} and {np.float32}')
    # check ndims
    if not rgb:
        raise NotImplementedError('output will always be RGB.')
    if img.ndim != 3:
        raise ValueError(f'image must have 3 dimensions, (H, W, C) if a numpy array, or (C, H, W) if a torch tensor, got shape: {img.shape}')
    # handle torch tensor
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().permute(1, 2, 0).numpy()
    # check image dtype
    if img.dtype not in (np.float32, np.uint8):
        raise ValueError(f'invalid image dtype: {img.dtype}, only supports: {np.uint8} and {np.float32}')
    # handle greyscale and rgba
    if img.shape[-1] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]
    # make sure we are RGB
    if img.shape[-1] != 3:
        raise ValueError(f'image has invalid number of channels, only supports 1 (grey), 3 (rgb) or 4 (rgba) channels, got shape: {img.shape}')
    # handle clipping
    if img.dtype == np.float32:
        img = np.clip(img, 0, 1)
        if dtype == np.uint8:
            img = (img * 255).astype(np.uint8)
    else:
        if dtype == np.float32:
            img = img.astype(np.float32) / 255
    # return image
    return img


def im_read(path_or_url, size: int = None, tensor=True) -> Union[np.ndarray, torch.Tensor]:
    """
    In general across this library we consider:
    - numpy images as having the shape (H, W, C).
    - torch/tensor images as having the shape (C, H, W).
    - images to be of dtype float32 [0, 1], except in save
      and load functions which might convert to uint8 [0, 255].
    """
    img = imageio.imread(path_or_url)
    img = im_to_numpy(img, dtype=np.uint8)
    # resize image
    if size is not None:
        img = PIL.Image.fromarray(img)
        img.thumbnail((size, size), PIL.Image.ANTIALIAS)
        img = np.array(img)
    # convert to float32
    img = np.float32(img) / 255
    if tensor:
        return torch.from_numpy(img).permute(2, 0, 1)
    return img


def im_show(img: Union[torch.Tensor, np.ndarray], figwidth=10):
    # normalise image
    img = im_to_numpy(img, dtype=np.uint8)
    # plot iamge
    H, W, C = img.shape
    fig, ax = plt.subplots(1, 1, figsize=(figwidth, figwidth * (H / W) + 0.25))
    ax.imshow(img)
    ax.set_axis_off()
    fig.tight_layout()
    plt.show()


# ========================================================================= #
# Callbacks                                                                 #
# ========================================================================= #


class VisualiseNCA(pl.Callback):

    def __init__(self, steps=(128, 256, 512, 1024, 2048), period=500, img_size=128, update_ratio=0.5, figwidth=10, figpadpx=8):
        self._period = period
        self._count = 0
        self._img_size = img_size
        self._figwidth = figwidth
        self._figpadpx = figpadpx
        self._update_ratio = update_ratio
        self._steps = {steps} if isinstance(steps, int) else set(steps)

    def on_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self._count += 1
        if self._count % self._period != 0:
            return
        # generate images
        with torch.no_grad():
            x = pl_module.nca.make_start_organisms(1, self._img_size, device=pl_module.device)
            images = []
            for i in range(max(self._steps)+1):
                # TODO: this is not general -- make generic NCA class?
                x, img = pl_module.nca.forward(x, update_ratio=self._update_ratio, return_img=True)
                # feed forward!
                if i in self._steps:
                    if len(images) > 0:
                        _, C, H, _ = img.shape
                        images.append(torch.ones(C, H, self._figpadpx, dtype=img.dtype, device=pl_module.device))
                    images.append(img[0])
        # visualise
        im_show(torch.cat(images, dim=-1), figwidth=self._figwidth)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
