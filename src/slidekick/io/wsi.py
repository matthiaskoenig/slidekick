import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
import zarr

from slidekick.console import console
from .czi import czi2tiff
from .metadata import Metadata
from .add_metadata import add_metadata


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

    # Pull channel names/colors from transformed OME-TIFF metadata if available
    if hasattr(metadata, "enrich_from_storage"):
        metadata.enrich_from_storage(overwrite=False)

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

    # Keep your tifffile.imread(..., aszarr=True) flow, but stop using .info
    store = tifffile.imread(str(image_path), aszarr=True, maxworkers=max_workers)
    root = zarr.open(store, mode="r")

    # If root is a single array
    if isinstance(root, zarr.Array):
        return {0: root}, image_path

    # Else root is a group: collect arrays by numeric keys
    arrays: dict[int, zarr.Array] = {}

    # Try array_keys() first (zarr v2)
    keys = []
    try:
        keys = list(root.array_keys())
    except Exception:
        pass

    # Fallback: generic keys() or probing "0","1",...
    if not keys:
        try:
            keys = list(root.keys())
        except Exception:
            keys = []

    if keys:
        for k in sorted(keys, key=lambda x: int(x) if str(x).isdigit() else float("inf")):
            arr = root.get(k)
            if arr is not None:
                idx = int(k) if str(k).isdigit() else len(arrays)
                arrays[idx] = arr
    else:
        i = 0
        while True:
            arr = root.get(str(i))
            if arr is None:
                break
            arrays[i] = arr
            i += 1

    # If still empty, fall back to TiffFile().series[0].aszarr()
    if not arrays:
        with tifffile.TiffFile(str(image_path)) as tf:
            s = tf.series[0]
            z = s.aszarr(maxworkers=max_workers)
            if isinstance(z, list):
                arrays = {i: z[i] for i in range(len(z))}
            else:
                arrays = {0: z}

    return arrays, image_path


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
