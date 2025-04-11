"""Napari viewer."""

from typing import List

import dask.array as da
import napari
import numpy as np


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
