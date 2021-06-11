from typing import Union

import imageio
import numpy as np
import PIL
import torch
from matplotlib import pyplot as plt


# ========================================================================= #
# Image Helper                                                              #
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


def im_show(img: Union[torch.Tensor, np.ndarray], figwidth: float = 10, title: str = None):
    # normalise image
    img = im_to_numpy(img, dtype=np.uint8)
    # plot iamge
    H, W, C = img.shape
    fig, ax = plt.subplots(1, 1, figsize=(figwidth, figwidth * (H / W) + (0.25 if title is None else 0.5)))
    ax.imshow(img)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()


_FILL_VALS = {
    np.dtype('uint8'): 255,
    np.dtype('float32'): 1.0,
}


def im_row(im_list, pad=8, border=False, vert=False, dtype=np.uint8):
    # normalise images & check
    (image, *images) = [im_to_numpy(img, dtype=dtype) for img in im_list]
    assert all(image.shape == img.shape for img in images), 'images do not have the same shapes'
    assert all(image.dtype == img.dtype for img in images), 'images do not have the same dtypes'
    # padding image
    H, W, C = image.shape
    pad_im = np.full([pad, W, C] if vert else [H, pad, C], dtype=image.dtype, fill_value=_FILL_VALS[image.dtype])
    # generate row
    row = [pad_im, image] if border else [image]
    for i, img in enumerate(images):
        row.extend([pad_im, img])
    row = [*row, pad_im] if border else row
    # concatenate
    return np.concatenate(row, axis=0 if vert else 1)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
