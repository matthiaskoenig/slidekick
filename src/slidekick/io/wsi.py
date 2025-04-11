import os
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import zarr

from slidekick.console import console
from slidekick.io.czi import czi2tiff


def read_wsi(image_path: Path, max_workers=os.cpu_count() - 1) -> dict[int, zarr.Array]:
    """Read image with tifffile library.

    @:return dictionary containing the resolution level as key and the zarr array of the image as value.
    """
    if image_path.suffix == ".czi":

        # FIXME: this is hardcoded for the CZI example
        czi_kwargs = dict(
            tile_size=(2048, 2048),
            subresolution_levels=[1, 2, 4],
            res_unit="µm"
        )
        czi_path = image_path
        image_path = Path(image_path).with_suffix(".tiff")

        if not image_path.exists():
            czi2tiff(czi_path, image_path, **czi_kwargs)
        else:
            console.print("CZI already converted to TIFF", style="warning")


    # read in zarr store
    store: tifffile.ZarrTiffStore = tifffile.imread(str(image_path), aszarr=True, maxworkers=max_workers)
    zarr_group = zarr.open(store, mode="r")  # zarr.core.Group or Array

    # find number of pyramidal levels
    for key, value in zarr_group.info.items:
        if key == "No. arrays":
            n_arrays = int(value)
            break

    # dictionary of pyramidal levels
    d: dict[int, zarr.Array] = {}
    for level in range(n_arrays):
        d[level] = zarr_group.get(level)

    return d


# import cv2
# from tifffile import TiffWriter

# def get_new_xy(y, x, level) -> Tuple[int, int]:
#     return round(y / 2 ** level), round(x / 2 ** level)


# def get_shape_for_level(shape_zero: Tuple[int, ...], level: int) -> Tuple[int, ...]:
#     if len(shape_zero) == 2:
#         y, x = shape_zero
#         new_y, new_x = get_new_xy(y, x, level)
#         return new_y, new_x
#
#     else:
#         y, x, c = shape_zero
#         new_y, new_x = get_new_xy(y, x, level)
#
#         return new_y, new_x, c


# def write_rois_to_ome_tiff(path: Path,
#                            image_level: int,
#                            image: np.ndarray,
#                            sub_res: int,
#                            tile_size: Tuple[int, int],
#                            level_zero_pixel_size: float,
#                            unit: str = "µm"
#                            ):
#     # remove zero level generator from dict
#
#     pixel_size = level_zero_pixel_size * 2 ** image_level
#
#     meta_data = {
#         "PhysicalSizeX": pixel_size,
#         "PhysicalSizeXUnit": unit,
#         "PhysicalSizeY": pixel_size,
#         "PhyiscalSizeYUnit": unit,
#     }
#
#
#     with TiffWriter(path, bigtiff=True) as tif:
#         options = dict(
#             photometric='rgb',
#             compression='jpeg',
#             resolutionunit='CENTIMETER',
#             maxworkers=os.cpu_count())
#
#         # writes the highest resolution
#         tif.write(
#             data=image,
#             tile=tile_size,
#             subifds=sub_res,
#             resolution=(1e4 / pixel_size, 1e4 / pixel_size),
#
#             # resolution=(1e4 / pixelsize, 1e4 / pixelsize),
#             metadata=meta_data,
#             **options
#         )
#
#         # write pyramid levels to the two subifds
#         # in production use resampling to generate sub-resolution images
#
#         for i in range(sub_res):
#             image = cv2.pyrDown(image)
#             mag = 2 ** (i + 1)
#             tif.write(
#                 data=image,
#                 subfiletype=1,
#                 resolution=(1e4 / pixel_size / mag, 1e4 / pixel_size / mag),
#                 **options,
#
#             )
#         # resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize), **options)
#         # add a thumbnail image as a separate series
#         # it is recognized by QuPath as an associated image
#
#         # thumbnail = (data[0, 0, ::8, ::8] >> 2).astype('uint8')
#
#         # tif.write(thumbnail, metadata={'Name': 'thumbnail'})
