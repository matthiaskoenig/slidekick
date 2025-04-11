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


if __name__ == "__main__":
    # Example for reading files
    from slidekick import DATA_PATH
    image_paths: list[Path] = [f for f in DATA_PATH.glob("*.*")]

    n_images = 0
    for image_path in image_paths:
        if image_path.suffix in {".ndpi", ".qptiff", ".tiff"}:
            # if image_path.suffix in {".qptiff"}:
            # if image_path.suffix in {".ndpi"}:
            # if image_path.suffix in {".czi"}:
            console.print(f"Importing image: '{image_path.name}'")
            image_data: dict[int, zarr.Array] = read_wsi(image_path)
            console.print(image_data)
            image_array: np.ndarray = np.array(image_data)

            n_images += 1

    console.print(f"Images imported: {n_images}")
