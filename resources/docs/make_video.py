
import imageio
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from nca.util.im import im_read
from nca.util.im import im_row
from nca.util.im import im_save


if __name__ == '__main__':

    vid_files = [
        '/Users/nmichlo/Desktop/active/cellular-autoencoder/out/2021-06-11_04-14-39/visualise_25000.mp4',  # microbe
        '/Users/nmichlo/Desktop/active/cellular-autoencoder/out/2021-06-11_16-12-30/visualise_10000.mp4',  # pattern
        '/Users/nmichlo/Desktop/active/cellular-autoencoder/out/2021-06-11_17-35-56/visualise_25000.mp4',  # large rocks
        '/Users/nmichlo/Desktop/active/cellular-autoencoder/out/2021-06-11_14-01-31/visualise_25000.mp4',  # small rocks
        '/Users/nmichlo/Desktop/active/cellular-autoencoder/out/2021-06-11_19-57-26/visualise_11000.mp4',  # wool
    ]

    scale, skip = 0.5, 5
    assert 30 % skip == 0

    with imageio.get_writer('out.gif', fps=30 // skip) as f:
        for i, frames in enumerate(tqdm(zip(*(imageio.get_reader(file) for file in vid_files)), desc='converting')):
            if i % skip == 0:
                f.append_data(np.array(ImageOps.scale(Image.fromarray(im_row(frames)), scale)))

    image_files = [
        '/Users/nmichlo/Desktop/active/cellular-autoencoder/data/textures/microbe.png',    # microbe
        '/Users/nmichlo/Desktop/active/cellular-autoencoder/data/textures/squiggles.jpg',  # pattern
        '/Users/nmichlo/Desktop/active/cellular-autoencoder/data/textures/rocks_large.jpg',# large rocks
        '/Users/nmichlo/Desktop/active/cellular-autoencoder/data/textures/rocks.jpg',      # small rocks
        '/Users/nmichlo/Desktop/active/cellular-autoencoder/data/textures/yarn.png',       # wool
    ]

    im_save(im_row([im_read(file, size=256, crop_square=True) for file in image_files]), 'out.png')

