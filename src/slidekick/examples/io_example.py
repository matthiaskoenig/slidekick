"""Using tifffile to read data"""

from pathlib import Path
from typing import List, Tuple
from xml.etree.ElementTree import XML

import dask.array as da
import napari

# from aicspylibczi import CziFile
# import bioformats
import zarr
from aicsimageio import AICSImage, readers
from tifffile import TiffFileError, ZarrStore, tiffcomment, tifffile

from zia import (
    TEST_FILES_PATHS,
    file_path_czi,
    file_path_czi2,
    file_path_npdi,
    file_path_tiff_exported2,
    file_path_tiff_test1,
)


def read_image_tifffile(image_path: Path) -> Tuple[XML, List[da.Array]]:
    """Read image with tifffile library."""
    if not image_path.exists():
        raise IOError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise IOError(f"Image file is not a file: {image_path}")

    print(f"Reading image with tifffile: {image_path.name}")
    try:
        # read in zarr store
        store = tifffile.imread(image_path, aszarr=True)
        group = zarr.open(store, mode="r")  # zarr.core.Group or Array

        datasets = group.attrs["multiscales"][0]["datasets"]

        # ome_xml = bioformats.get_omexml_metadata(file_path)
        # print(grp.attrs)
        # layers_n = 3
        # layers = {f"layer{x}":[] for x in range(layers_n)}
        # for d in datasets:
        #    data = da.from_zarr(store, component=d["path"])
        #    for layer_n in range(layers_n):
        #        layers[f"layer{layer_n}"].append(data[layer_n,:,:])

        # convert in list of dask arrays
        data = [da.from_zarr(store, component=d["path"]) for d in datasets]

        # data = [da.moveaxis(da.from_zarr(store, component=d["path"]),0,2) for d in datasets]

        return data
    except TiffFileError as e:
        img = readers.czi_reader.CziReader(image_path)
        print(f"Error reading file: {image_path.name}")

        # czi = CziFile(file_path)
        # dimensions = czi.get_dims_shape()
        # print(dimensions)

        # img = AICSImage(file_path, reconstruct_mosaic=False)  # selects the first scene found

        properties = [
            img.dims,  # returns a Dimensions object
            img.dims.order,  # returns string "TCZYX"
            img.dims.X,  # returns size of X dimension
            img.shape,  # returns tuple of dimension sizes in TCZYX order
            img.scenes,  # returns total number of pixels
        ]

        print(properties)
        # print(img.get_image_data("TCZYXS"))

        return img.get_image_data("TCZYXS")

        # raise e


if __name__ == "__main__":
    # napari_view_ndpi(file_path_npdi,)

    # napari_view(file_path_tiff_exported2,)

    # napari_view(file_path_tiff_test1)

    napari_view_ndpi(file_path_czi2)


# for path in TEST_FILES_PATHS:
#    store = read_file(path)


from pprint import pprint

import zarr
from napari_lazy_openslide import OpenSlideStore
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from tifffile import imread

from zia import DATA_PATH


if __name__ == "__main__":
    # print(OME_TIFF_EXAMPLE)
    # store = OpenSlideStore(OME_TIFF_EXAMPLE)
    # grp = zarr.open(store, mode="r")
    # with napari.gui_qt():
    #    viewer = napari.Viewer()
    #    viewer.open(OME_TIFF_EXAMPLE)

    # OpenSlideStore(OME_TIFF_EXAMPLE)
    file_path_luca = DATA_PATH / "LuCa-7color_Scan1.ome.tiff"
    file_path_npdi = DATA_PATH / "LQF2_LM_HE_PVL.ndpi"
    file_path_lqf2 = DATA_PATH / "LQF2_RM_HE_PVL2.ome.tif"
    file_path_czi = DATA_PATH / "RH0422_1x2.5.czi"

    file_path_tiff_test1 = DATA_PATH / "test_001.tif"
    file_path_tiff_test1_exported = DATA_PATH / "test_001_exported.ome.tif"
    file_path_tiff_test1_exported2 = DATA_PATH / "test_001_exported2.ome.tif"

    for file_path in [
        file_path_lqf2,
        file_path_npdi,
        file_path_tiff_test1,
        file_path_czi,
        file_path_tiff_test1_exported2,
    ]:
        try:
            s = OpenSlideStore(file_path)
            grp = zarr.open(s, mode="r")
            # The OpenSlideStore implements the multiscales extension
            # https://forum.image.sc/t/multiscale-arrays-v0-1/37930
            pprint(s._slide.properties)
            datasets = grp.attrs["multiscales"][0]["datasets"]
            pyramid = [grp.get(d["path"]) for d in datasets]
            print(pyramid)
        except OpenSlideUnsupportedFormatError:
            print(f"OpenSlideStore failed for {file_path}")
            s = None

    # imread(file_path_npdi)
    # print(file_path)
    slide = OpenSlide(file_path_tiff_test1_exported)
    slide1 = OpenSlide(file_path_tiff_test1_exported2)
    slide2 = OpenSlide(file_path_tiff_test1)

    for s in [slide, slide1, slide2, file_path_npdi]:
        # slide1 =OpenSlide(file_path_npdi)
        print(s.level_dimensions)
        print(s.dimensions)

    slide1 = OpenSlide(file_path_npdi)

    print(slide.level_dimensions)
    print(slide.dimensions)

    # with napari.gui_qt():
    #    viewer = napari.Viewer()
    #    viewer.open(OME_TIFF_EXAMPLE)
    #    viewer.add_image(grp["0"], name="0")
    #    viewer.add_image(grp["1"], name="1")
    #    viewer.add_image(grp["2"], name="2")
