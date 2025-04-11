import numpy as np
import zarr

from slidekick.console import console
from slidekick.io.wsi import read_wsi
from slidekick.visualization.napari_viewer import view_wsi

if __name__ == "__main__":
    from slidekick import DATA_PATH

    # read image
    # image_path = DATA_PATH / "SIM-22-034_4plex.qptiff"
    image_path = DATA_PATH / "NOR-021_CYP1A2.ndpi"
    image_data: dict[int, zarr.Array] = read_wsi(image_path)
    console.print(image_data)
    image_array: np.ndarray = np.array(image_data)


    view_wsi(image_array)

