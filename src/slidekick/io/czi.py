import os
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
import tifffile as tiff


def czi2tiff(
    czi_path: Path,
    ometiff_path: Path,
    tile_size: Tuple[int, int],
    subresolution_levels: List[int],
    res_unit: str = "µm",
):
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
    if (ots[0] % 16) or (ots[1] % 16):
        raise ValueError("Tile size must be multiple of 16")

    # --- Probe CZI to distinguish true RGB vs multiplex/grayscale channels ---
    with pyczi.open_czi(str(czi_path)) as czi_img:
        xstart, xend, ystart, yend = get_size(czi_img)
        ph = min(32, yend - ystart)
        pw = min(32, xend - xstart)
        probe = czi_img.read(
            plane={"TILES": 0, "Z": 0, "C": 0},
            roi=(xstart, ystart, pw, ph),
            background_pixel=(255, 255, 255),
            zoom=1.0,
        )
        probe = np.asarray(probe).squeeze()
        is_true_rgb = (probe.ndim == 3 and probe.shape[-1] == 3 and probe.dtype == np.uint8)

    if not is_true_rgb:
        # ---------- Fluorescence/multiplex path: write lossless OME-TIFF (YXC) ----------
        with pyczi.open_czi(str(czi_path)) as czi_img:
            xstart, xend, ystart, yend = get_size(czi_img)
            H = yend - ystart
            W = xend - xstart

            # Count channels by probing C until empty/failure
            n_ch = 0
            for c in range(256):
                try:
                    s = czi_img.read(
                        plane={"TILES": 0, "Z": 0, "C": c},
                        roi=(xstart, ystart, 1, 1),
                        background_pixel=(0, 0, 0),
                        zoom=1.0,
                    )
                    a = np.asarray(s)
                    if a.size == 0:
                        break
                    n_ch += 1
                except Exception:
                    break
            n_ch = max(1, n_ch)

            # Pixel size metadata (optional)
            try:
                meta_data = translate_to_metadata(czi_img)
                psx, psy = get_pixel_size_from_meta(meta_data)
                res = (psy.to(res_unit).magnitude, psx.to(res_unit).magnitude)
                resunit = "MICROMETER"
            except Exception:
                res, resunit = None, None

            # Read full plane per channel and stack to YXC
            ch0 = czi_img.read(
                plane={"TILES": 0, "Z": 0, "C": 0},
                roi=(xstart, ystart, W, H),
                background_pixel=(0, 0, 0),
                zoom=1.0,
            )
            ch0 = np.asarray(ch0).squeeze()
            if ch0.ndim == 3 and ch0.shape[-1] == 1:
                ch0 = ch0[..., 0]
            data_yxc = np.empty((H, W, n_ch), dtype=ch0.dtype)
            data_yxc[..., 0] = ch0

            for c in range(1, n_ch):
                tile = czi_img.read(
                    plane={"TILES": 0, "Z": 0, "C": c},
                    roi=(xstart, ystart, W, H),
                    background_pixel=(0, 0, 0),
                    zoom=1.0,
                )
                arr = np.asarray(tile).squeeze()
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr[..., 0]
                data_yxc[..., c] = arr

        imw = dict(
            bigtiff=True,
            ome=True,
            metadata={"axes": "YXC"},
            tile=ots,            # tiles over YX
            compression="zlib",  # lossless for fluorescence
            maxworkers=os.cpu_count(),
        )
        if res is not None:
            imw.update(dict(resolution=res, resolutionunit=resunit))
        tiff.imwrite(str(ometiff_path), data_yxc, **imw)
        return

    # ---------- Legacy RGB path (tiled JPEG pyramid) ----------
    with (pyczi.open_czi(str(czi_path)) as czi_img, Progress() as progress):
        xstart, xend, ystart, yend = get_size(czi_img)
        th, tw = ots

        x_coords = np.arange(xstart, xend, tw)
        y_coords = np.arange(ystart, yend, th)
        n_tiles = len(x_coords) * len(y_coords)

        h, w = yend - ystart, xend - xstart

        meta_data = translate_to_metadata(czi_img)
        psx, psy = get_pixel_size_from_meta(meta_data)

        task = progress.add_task(
            description=f"[green]Converting {czi_path.name} to ome-tiff...",
            total=(len(subresolution_levels) + 1) * n_tiles,
        )
        sub_task = progress.add_task(description="[cyan]Writing resolution level 0...", total=n_tiles)

        with TiffWriter(ometiff_path, bigtiff=True) as tif:
            options = dict(
                photometric="rgb",
                compression="jpeg",
                resolutionunit="MICROMETER",
                maxworkers=os.cpu_count(),
            )

            # Highest resolution
            tif.write(
                data=load_image(czi_img, (x_coords, y_coords), 0, ots, progress, (task, sub_task)),
                tile=ots,
                shape=(h, w, 3),
                subifds=subresolution_levels,
                dtype=np.uint8,
                resolution=(psy.to(res_unit).magnitude, psx.to(res_unit).magnitude),
                metadata=meta_data,
                **options,
            )
            progress.update(task, advance=1)

            # Pyramid levels
            for level in subresolution_levels:
                progress.reset(sub_task, total=n_tiles, description=f"[cyan]Writing resolution level {level}...")
                mag = 2 ** level
                ts = (round(th / mag), round(tw / mag))

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
        meta_data["NominalMagnification"] = czi_image.metadata["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"]["NominalMagnification"]
    elif "ImageScaling" in czi_image.metadata["ImageDocument"]["Metadata"]:
        image_scaling = czi_image.metadata["ImageDocument"]["Metadata"]["ImageScaling"]
        magnifications = [float(sc["@Magnification"]) for sc in image_scaling["ScalingComponent"]]
        mag = np.prod(magnifications)
        meta_data["NominalMagnification"] = str(round(mag))
    else:
        print(
            "No magnification found in metadata. Setting magnification to 1."
            "Execution will continue. Please ensure that this is the expected behavior."
        )
        meta_data["NominalMagnification"] = str(f"{1}")

    distances = czi_image.metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
    for dist in distances:
        for id_ in ["X", "Y"]:
            if dist["@Id"] == id_:
                value = ureg.Quantity(float(dist["Value"]), "m")
                quant = value.to(dist["DefaultUnitFormat"])
                meta_data[f"PhysicalSize{id_}"] = quant.magnitude
                meta_data[f"PhysicalSizeUnit{id_}"] = format(quant.units, "~H")

    return meta_data


def load_image(
    czi_img: CziReader,
    coords: Tuple[np.ndarray, np.ndarray],
    res_level: int,
    ts: Tuple[int, int],
    progress: Progress,
    tasks: Tuple[TaskID, TaskID],
):
    """Legacy RGB tile generator. Yields (th, tw, 3) uint8 tiles."""
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

            yield (
                czi_img.read(
                    plane={"TILES": 0, "Z": 0, "C": 0},
                    roi=(x_coord, y_coord, e_w, e_h),
                    background_pixel=(255, 255, 255),
                    zoom=1 / fac,
                )
                .squeeze()
                .astype(np.uint8)
            )


def get_size(czi_img):
    xstart, xend = czi_img.total_bounding_box["X"]
    ystart, yend = czi_img.total_bounding_box["Y"]
    return xstart, xend, ystart, yend


def get_pixel_size_from_meta(meta_data) -> Tuple[pint.Quantity, pint.Quantity]:
    ureg = UnitRegistry()
    x_size = ureg.Quantity(meta_data["PhysicalSizeX"], meta_data["PhysicalSizeUnitX"])
    y_size = ureg.Quantity(meta_data["PhysicalSizeY"], meta_data["PhysicalSizeUnitY"])
    return x_size, y_size
