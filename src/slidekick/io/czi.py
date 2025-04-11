import os.path
from pathlib import Path
from typing import Tuple, List

import lxml.etree as ET
import numpy as np
import pint
import pylibCZIrw.czi as pyczi
from pint import UnitRegistry
from pylibCZIrw.czi import CziReader
from rich.progress import Progress, TaskID
from tifffile import TiffWriter

def czi2tiff(czi_path: Path, ometiff_path: Path, tile_size: Tuple[int, int], subresolution_levels: List[int], res_unit: str = "µm"):
    """Converts a czi file to the ome-tiff format.

    The image is written in a pyramidal fashion. Metadata is supported as far as
    the tiffile library allows. The magnification is unfortunately not written.

    @param czi_path: path to the czi file
    @param ometiff_path: path to the target ome tiff file
    @param tile_size: the size of the tiles in the target array. All elements must be multiples of 16
    @param subresolution_levels: the resolution levels for the pyramid levels to write (factor is 2**subresolution_level)
    @param res_unit: the resolution unit to write to metadata.
    """
    ots = tile_size

    if (ots[0] % 16 or ots[1] % 16):
        raise ValueError("Tile size must be multiple of 16")

    with (pyczi.open_czi(str(czi_path)) as czi_img, Progress() as progress):

        xstart, xend, ystart, yend = get_size(czi_img)
        th, tw = ots

        x_coords = np.arange(xstart, xend, tw)
        y_coords = np.arange(ystart, yend, th)

        n_tiles = len(x_coords) * len(y_coords)

        h, w = yend - ystart, xend - xstart

        meta_data = translate_to_metadata(czi_img)

        psx, psy = get_pixel_size_from_meta(meta_data)

        task = progress.add_task(description=f"[green]Converting {czi_path.name} to ome-tiff...", total=(len(subresolution_levels) + 1) * n_tiles)
        sub_task = progress.add_task(description="[cyan]Writing resolution level 0...", total=n_tiles)

        with TiffWriter(ometiff_path, bigtiff=True) as tif:
            options = dict(
                photometric='rgb',
                compression='jpeg',
                resolutionunit='MICROMETER',
                maxworkers=os.cpu_count())

            # writes the highest resolution
            tif.write(
                data=load_image(czi_img, (x_coords, y_coords), 0, ots, progress, (task, sub_task)),
                tile=ots,
                shape=(h, w, 3),
                subifds=subresolution_levels,
                dtype=np.uint8,
                resolution=(psy.to(res_unit).magnitude, psx.to(res_unit).magnitude),
                metadata=meta_data,
                **options
            )

            progress.update(task, advance=1)


            # write pyramid levels to the two subifds
            # in production use resampling to generate sub-resolution images

            for level in subresolution_levels:
                progress.reset(sub_task, total=n_tiles, description=f"[cyan]Writing resolution level {level}...")
                mag = 2 ** level

                ts = tuple([round(k / mag) for k in ots])

                tif.write(
                    data=load_image(czi_img, (x_coords, y_coords), level, ots, progress, (task, sub_task)),
                    tile=ts,
                    subfiletype=1,
                    shape=(round(h / mag), round(w / mag), 3),
                    resolution=((psy / mag).to(res_unit).magnitude, (psx / mag).to(res_unit).magnitude),
                    dtype=np.uint8,
                    **options,
                )

                progress.update(task, advance=1)

            progress.remove_task(sub_task)


def translate_czimetadata_to_ome_meta(czi_image: CziReader):
    # Parse template and generate transform function
    template = ET.parse(r"czi-to-ome-xslt\xslt\czi-to-ome.xsl")
    transformer = ET.XSLT(template)

    # Parse CZI XML
    czixml = ET.fromstring(czi_image.raw_metadata)

    # Transform
    omexml = transformer(czixml)

    return omexml


def translate_to_metadata(czi_image: CziReader):
    ureg = UnitRegistry()
    meta_data = {}

    if "Instrument" in czi_image.metadata["ImageDocument"]["Metadata"]["Information"]:
        # Extract nominal magnification from the metadata.
        meta_data["NominalMagnification"] = czi_image.metadata["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"]["NominalMagnification"]

    elif "ImageScaling" in czi_image.metadata["ImageDocument"]["Metadata"]:
        # Calculate the magnification from the ImageScaling MetaData.
        image_scaling = czi_image.metadata["ImageDocument"]["Metadata"]["ImageScaling"]

        magnifications = [float(sc["@Magnification"]) for sc in image_scaling["ScalingComponent"]]
        mag = np.prod(magnifications)

        meta_data["NominalMagnification"] = str(round(mag))

    else:
        # Some files may not have the magnification in the metadata. In this case, we set it to 1.
        print("No magnification found in metadata. Setting magnification to 1." +
              "Execution will continue. Please ensure that this is the expected behavior.")
        meta_data["NominalMagnification"] = str(f"{1}")

    distances = czi_image.metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]

    for dist in distances:
        for id in ["X", "Y"]:
            if dist["@Id"] == id:
                value = ureg.Quantity(float(dist["Value"]), "m")
                quant = value.to(dist["DefaultUnitFormat"])

                meta_data[f"PhysicalSize{id}"] = quant.magnitude
                meta_data[f"PhysicalSizeUnit{id}"] = format(quant.units, "~H")

    return meta_data


def load_image(czi_img: CziReader, coords: Tuple[np.ndarray, np.ndarray], res_level: int, ts: Tuple[int, int], progress: Progress,
               tasks: Tuple[TaskID, TaskID]) -> np.ndarray:
    fac = 2 ** res_level

    task, level_task = tasks
    th, tw = ts
    xstart, xend, ystart, yend = get_size(czi_img)

    x_coords, y_coords = coords

    for y_coord in y_coords:
        for x_coord in x_coords:
            progress.update(task, advance=1)
            progress.update(level_task, advance=1)

            e_w = min(tw, xend - x_coord)
            e_h = min(th, yend - y_coord)

            yield czi_img.read(
                plane={"TILES": 0, "Z": 0, "C": 0},
                roi=(x_coord, y_coord, e_w, e_h),
                background_pixel=(255, 255, 255),
                zoom=1 / fac
            ).squeeze().astype(np.uint8)


def get_size(czi_img):
    xstart, xend = czi_img.total_bounding_box["X"]
    ystart, yend = czi_img.total_bounding_box["Y"]

    return xstart, xend, ystart, yend


def get_pixel_size_from_meta(meta_data) -> Tuple[pint.Quantity, pint.Quantity]:
    ureg = UnitRegistry()
    x_size = ureg.Quantity(meta_data["PhysicalSizeX"], meta_data["PhysicalSizeUnitX"])
    y_size = ureg.Quantity(meta_data["PhysicalSizeY"], meta_data["PhysicalSizeUnitY"])
    return x_size, y_size
