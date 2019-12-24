import logging
import os

from csbdeep.utils import plot_history

import imageio
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2VConfig, N2V
import numpy as np

from utoolbox.io.dataset import SmartSpimDataset

logger = logging.getLogger(__name__)


def train(images, patch_shape=(96, 96), ratio=0.7, name="untitled", model_dir="."):
    # create data generator object
    datagen = N2V_DataGenerator()

    # TODO sample in between timeseries
    # patch additional axes
    _images = []
    for image in images:
        _images.append(image[np.newaxis, ..., np.newaxis])
    images = _images 

    patches = datagen.generate_patches_from_list(images, shape=patch_shape)
    logger.info(f"patch shape: {patches.shape}")
    n_patches = patches.shape[0]
    logger.info(f"{n_patches} patches generated")

    # split training set and validation set
    i = int(n_patches * ratio)
    X, X_val = patches[:i], patches[i:]

    # create training config
    config = N2VConfig(
        X,
        unet_kern_size=3,
        train_steps_per_epoch=100,
        train_epochs=100,
        train_loss="mse",
        batch_norm=True,
        train_batch_size=4,
        n2v_perc_pix=1.6,
        n2v_patch_shape=patch_shape,
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius=5,
    )

    model = N2V(config=config, name=name, basedir=model_dir)

    # train and save the model
    history = model.train(X, X_val)
    model.export_TF()

    plot_history(history, ["loss", "val_loss"])


def infer(images, name="untitled", model_dir="."):
    model = N2V(config=None, name=name, basedir=model_dir)

    for image in images:
        yield model.predict(image, axes="YX")


def run(src_dir, dst_dir):
    ds = SmartSpimDataset(src_dir)
    print(ds.inventory)

    for ch, ds_c in ds.groupby("channel"):
        if ch != "642":
            continue

        images = [np.squeeze(ds[uuid].compute()) for _, uuid in ds_c.iteritems()]

        break

    train(images, name="642")

    try:
        dst_dir = os.path.join(dst_dir, "642_n2v")
        os.makedirs(dst_dir)
    except FileExistsError:
        pass
    for i, image in enumerate(infer(images, name="642")):
        imageio.imwrite(os.path.join(dst_dir, f"tile_{i:04d}.tif"), image)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    run("C:/Users/Andy/Desktop/20191217_16_50_20_cerebellum_tile", "_debug")

