import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
import zarr

from slidekick.console import console
from slidekick.io.czi import czi2tiff
from slidekick.metadata.metadata import Metadata
from slidekick.metadata import add_metadata


def import_wsi(image_path: Path, annotate: bool = False) -> Tuple[dict[int, zarr.Array], Metadata]:
    """
    Import whole-slide image and return data and metadata.

    If annotate=True, interactively annotate metadata (image type and stains).
    """
    image_data, stored_path = read_wsi(image_path)

    # Initialize metadata object first
    metadata = Metadata(
        path_original=image_path,
        path_storage=stored_path,
    )

    if annotate:
        add_metadata(image_data, metadata)

    return image_data, metadata


def read_wsi(image_path: Path, max_workers=os.cpu_count() - 1) -> Tuple[dict[int, zarr.Array], Path]:
    if image_path.suffix == ".czi":
        czi_kwargs = dict(tile_size=(2048, 2048), subresolution_levels=[1, 2, 4], res_unit="µm")
        czi_path = image_path
        image_path = image_path.with_suffix(".tiff")

        if not image_path.exists():
            czi2tiff(czi_path, image_path, **czi_kwargs)
        else:
            console.print("CZI already converted to TIFF", style="warning")

    store = tifffile.imread(str(image_path), aszarr=True, maxworkers=max_workers)
    zarr_group = zarr.open(store, mode="r")

    for key, value in zarr_group.info.items:
        if key == "No. arrays":
            n_arrays = int(value)
            break

    d: dict[int, zarr.Array] = {level: zarr_group.get(level) for level in range(n_arrays)}

    return d, image_path

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
            image_data, meta_data = read_wsi(image_path)
            console.print(image_data)
            console.print(meta_data)
            image_array: np.ndarray = np.array(image_data)

            n_images += 1

    console.print(f"Images imported: {n_images}")
