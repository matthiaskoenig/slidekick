"""Importing WSI images."""
from pathlib import Path
import numpy as np
import zarr

from slidekick.console import console
from slidekick.io.wsi import read_wsi


def wsi_import(image_path: Path) -> None:
    console.print(f"Importing image: '{image_path.name}'")
    image_data: dict[int, zarr.Array] = read_wsi(image_path)
    console.print(image_data)
    image_array: np.ndarray = np.array(image_data)



if __name__ == "__main__":
    from slidekick import DATA_PATH
    image_paths: list[Path] = [f for f in DATA_PATH.glob("*.*")]

    n_images = 0
    for image_path in image_paths:
        if image_path.suffix in {".ndpi", ".qptiff", ".tiff"}:
        # if image_path.suffix in {".qptiff"}:
        # if image_path.suffix in {".ndpi"}:
        # if image_path.suffix in {".czi"}:
            wsi_import(image_path)
            n_images+= 1
        
    console.print(f"Images imported: {n_images}")