from collections import defaultdict
from itertools import product
import logging
import os
from pprint import pprint

import coloredlogs
from dask import delayed
from dask.cache import Cache
from dask.diagnostics import ProgressBar
from dask.distributed import Client, fire_and_forget
import numpy as np

from utoolbox.io.dataset.mm import MicroManagerV2Dataset


def create_dst_dir(name, root_dir="S:/Andy/_debug/HD91"):
    path = os.path.join(root_dir, name)
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path


def load_valid_tiles(ds, save=False):
    import imageio

    if save:
        dst_dir = create_dst_dir("mip")

    # generate thumbnails
    images = []
    for j, (_, ds_x) in enumerate(ds.groupby("tile_y")):
        for i, (_, tile) in enumerate(ds_x.groupby("tile_x")):
            print(f".. iter (i: {i}, j: {j})")

            uuid = tile.loc["488_1X"].values[0]
            # ignore missing tiles
            if not uuid:
                continue

            if save:
                data = ds[uuid].max(axis=0)
                dst_path = os.path.join(dst_dir, f"tile-{i:03d}-{j:03d}_mip.tif")
                # delayed(imageio.imwrite)(dst_path, data)

                future = client.submit(imageio.imwrite, dst_path, data)
                fire_and_forget(future)

            images.append(((j, i), uuid))

    #   2 3 4
    # 6
    # 7
    #
    # generate neighbor linkage
    links = dict()
    for (apos, auuid), (bpos, buuid) in product(images[:-1], images[1:]):
        if apos == bpos:
            continue

        if bpos < apos:
            apos, bpos = bpos, apos
            auuid, buuid = buuid, auuid
        print(f"{apos} <> {bpos}")

        if (apos, bpos) in links:
            print(".. duplicate")
            continue
        else:
            aj, ai = apos
            bj, bi = bpos
            if ((aj - 1 == bj or aj + 1 == bj) and (ai == bi)) or (
                (ai - 1 == bi or ai + 1 == bi) and (aj == bj)
            ):
                print(".. NEW NEIGHBOR")
                links[(apos, bpos)] = (auuid, buuid)
            else:
                print(".. not neighbor")

    return images, links


def calculate_link_shifts(ds, links, overlap=0.1):
    from math import floor, ceil

    import matplotlib.pyplot as plt
    from skimage.feature import register_translation
    from scipy import ndimage as ndi

    for (apos, bpos), (auuid, buuid) in links.items():
        print(f"{apos} <> {bpos}")

        aj, ai = apos
        bj, bi = bpos

        ashape, bshape = ds[auuid].shape, ds[buuid].shape

        for ratio in (overlap, 1 - overlap):
            if ai == bi:
                # overlap on Y
                ar, br = ceil(ashape[1] * ratio), floor(bshape[1] * (1 - ratio))
                if ratio < 0.5:
                    ablk = ds[auuid][:, :ar, :]
                    bblk = ds[buuid][:, br:, :]
                else:
                    ablk = ds[auuid][:, ar:, :]
                    bblk = ds[buuid][:, :br, :]
            else:
                # overlap on X
                ar, br = ceil(ashape[2] * ratio), floor(bshape[2] * (1 - ratio))
                if ratio < 0.5:
                    ablk = ds[auuid][..., :ar]
                    bblk = ds[buuid][..., br:]
                else:
                    ablk = ds[auuid][..., ar:]
                    bblk = ds[buuid][..., :br]

            with ProgressBar():
                amip, bmip = ablk.max(axis=0), bblk.max(axis=0)
                amip, bmip = amip.compute(), bmip.compute()
                print(amip.shape)
                print(bmip.shape)

            shift, error, diffphase = register_translation(amip, bmip)
            print(error)

            amip2 = ndi.shift(amip, -shift / 2, order=2, mode="constant", cval=0)
            bmip2 = ndi.shift(bmip, shift / 2.0, order=2, mode="constant", cval=0)

            result = np.maximum(amip2, bmip2)

            ny, nx = amip2.shape

            ## preview overlap region
            # if ny <= nx:
            #    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
            # else:
            #    fig, ax = plt.subplots(1, 3, figsize=(12, 12))
            # ax[0].imshow(amip, cmap="gray")
            # ax[1].imshow(bmip, cmap="gray")
            # ax[2].imshow(result, cmap="gray")
            # plt.show(block=False)
            # plt.waitforbuttonpress()

            with ProgressBar():
                amip = ds[auuid].max(axis=0).compute()
                bmip = ds[buuid].max(axis=0).compute()

            ny, nx = amip.shape

            amip2 = ndi.shift(
                amip, (0, -(1 - overlap) * nx / 2.0), order=2, mode="constant", cval=0
            )
            bmip2 = ndi.shift(
                bmip, (0, (1 - overlap) * nx / 2.0), order=2, mode="constant", cval=0
            )

            amip2 = ndi.shift(amip2, -shift / 2, order=2, mode="constant", cval=0)
            bmip2 = ndi.shift(bmip2, shift / 2.0, order=2, mode="constant", cval=0)

            result = np.maximum(amip2, bmip2)

            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(result, cmap="gray")
            plt.show(block=False)
            plt.waitforbuttonpress()


if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    client = Client("10.109.20.6:8786")
    print(client)
    print()

    cache = Cache(2e9)
    cache.register()

    src_dir = "Y:/charm/20191015_Clarity_brain_HD91_RCCS_1"
    ds = MicroManagerV2Dataset(src_dir)

    print("== dataset inventory ==")
    print(ds.inventory)

    images, links = load_valid_tiles(ds, save=True)
    # calculate_link_shifts(ds, links)

    client.close()
