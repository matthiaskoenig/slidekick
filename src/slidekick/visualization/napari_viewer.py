"""Napari viewer.

Visualization is performed using https://napari.org/stable/
napari: a fast, interactive viewer for multi-dimensional images in Python

https://github.com/AstraZeneca/napari-wsi
"""

from typing import List

import dask.array as da
import napari
import numpy as np
import zarr

from slidekick.console import console


def view_wsi(data: List[da.Array]) -> None:
    """View WSI in internal format..

    This is starting napari and blocking.
    """
    viewer = napari.Viewer()
    viewer.add_image(data)
    napari.run()


def view_czi_data(data: np.ndarray, channel_names: List[str]) -> None:
    """View CZI image in napari.

    This is starting napari and blocking.

    Each channel in a multichannel image can be displayed as an individual layer by
    using the channel_axis argument in viewer.add_image(). All the rest of the
    arguments to viewer.add_image() (e.g. name, colormap, contrast_limit) can take
    the form of a list of the same size as the number of channels.
    """
    viewer = napari.Viewer()
    viewer.add_image(
        data,
        channel_axis=2,
        name=channel_names,
    )
    napari.run()

if __name__ == "__main__":
    from slidekick import DATA_PATH
    from slidekick.io import wsi

    # read image
    image_path = DATA_PATH / "SIM-22-034_4plex.qptiff"
    # image_path = DATA_PATH / "NOR-021_CYP1A2.ndpi"
    image_data: dict[int, zarr.Array] = wsi.read_wsi(image_path)
    console.print(image_data)
    image_array: np.ndarray = np.array(image_data)

    view_wsi(image_data[4])

