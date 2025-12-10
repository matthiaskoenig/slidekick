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
        # Number of logical channels from the C dimension
        bbox = czi_img.total_bounding_box
        c_start, c_end = bbox.get("C", (0, 1))
        n_channels_c = max(1, int(c_end - c_start))

        xstart, xend, ystart, yend = get_size(czi_img)
        ph = min(32, yend - ystart)
        pw = min(32, xend - xstart)

        # Small probe from the first C plane
        probe = czi_img.read(
            plane={"TILES": 0, "Z": 0, "C": c_start},
            roi=(xstart, ystart, pw, ph),
            background_pixel=(255, 255, 255),
            zoom=1.0,
        )
        probe = np.asarray(probe)

        # Metadata-driven check: true color channels are stored as BGR/RGB pixel types
        pixel_types = getattr(czi_img, "pixel_types", None)
        is_color_pixel_type = False
        if isinstance(pixel_types, dict):
            for pt in pixel_types.values():
                if not pt:
                    continue
                pt_str = str(pt).lower()
                if "bgr" in pt_str or "rgb" in pt_str:
                    is_color_pixel_type = True
                    break

        # Shape/dtype heuristic as a fallback
        is_color_probe = (
                probe.ndim == 3
                and probe.shape[-1] == 3
                and probe.dtype == np.uint8
        )

        # Treat as true RGB only if there is a single logical C channel
        # and the data really looks like BGR/RGB
        is_true_rgb = (n_channels_c == 1) and (is_color_pixel_type or is_color_probe)

    if not is_true_rgb:
        # ---------- Fluorescence/multiplex path: pyramidal lossless OME-TIFF (YXC) ----------
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

            # Channel names from raw CZI XML
            channel_names = extract_channel_names_from_czi(czi_img, n_ch)

            # Pixel size metadata (optional)
            try:
                meta_data = translate_to_metadata(czi_img)
                psx, psy = get_pixel_size_from_meta(meta_data)
                psx_um = psx.to(res_unit).magnitude
                psy_um = psy.to(res_unit).magnitude
                res = (psy_um, psx_um)
                resunit = "MICROMETER"
            except Exception:
                psx_um = psy_um = None
                res = resunit = None

            # Read full plane per channel and stack to YXC (internal layout)
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

            # Convert to CYX for OME so each logical channel becomes a separate OME Channel
            data_cyx = np.moveaxis(data_yxc, -1, 0)  # (C, Y, X)

            # Build OME-compatible metadata dict (this carries channel names)
            # axes='CYX' + data shape (C, Y, X) -> SizeC = n_ch, SamplesPerPixel = 1
            ome_metadata = {
                "axes": "CYX",
                "Channel": {"Name": channel_names},
            }

            if psx_um is not None and psy_um is not None:
                ome_metadata.update(
                    {
                        "PhysicalSizeX": psx_um,
                        "PhysicalSizeXUnit": res_unit,
                        "PhysicalSizeY": psy_um,
                        "PhysicalSizeYUnit": res_unit,
                    }
                )

            # TIFF writing options: tiled, lossless compression for fluorescence
            # For CYX, we let tifffile choose planar configuration; no explicit planarconfig.
            options = dict(
                tile=ots,  # tiles over YX
                compression="zlib",  # lossless
                photometric="minisblack",
                maxworkers=os.cpu_count(),
            )
            if resunit is not None:
                options["resolutionunit"] = resunit

            # Write base level + pyramid as OME-TIFF
            with TiffWriter(str(ometiff_path), bigtiff=True, ome=True) as tif:
                # Base (full-resolution) CYX plane, reserve space for SubIFDs
                tif.write(
                    data_cyx,
                    subifds=len(subresolution_levels),
                    dtype=data_cyx.dtype,
                    shape=data_cyx.shape,
                    resolution=res,
                    metadata=ome_metadata,
                    **options,
                )

                # Pyramid levels written as SubIFDs (simple decimation)
                for level in subresolution_levels:
                    mag = 2 ** level
                    # Downsample Y and X, keep C unchanged
                    level_data = data_cyx[:, ::mag, ::mag]
                    if res is not None:
                        level_res = (res[0] / mag, res[1] / mag)
                    else:
                        level_res = None

                    tif.write(
                        level_data,
                        subfiletype=1,  # reduced-resolution
                        dtype=level_data.dtype,
                        shape=level_data.shape,
                        resolution=level_res,
                        metadata=None,
                        **options,
                    )
            return

    # ---------- RGB path (tiled JPEG pyramidal OME-TIFF) ----------
    with (pyczi.open_czi(str(czi_path)) as czi_img, Progress() as progress):
        xstart, xend, ystart, yend = get_size(czi_img)
        th, tw = ots

        x_coords = np.arange(xstart, xend, tw)
        y_coords = np.arange(ystart, yend, th)
        n_tiles = len(x_coords) * len(y_coords)

        h, w = yend - ystart, xend - xstart

        meta_data = translate_to_metadata(czi_img)
        psx, psy = get_pixel_size_from_meta(meta_data)

        # Simple OME metadata for true RGB (single logical channel, 3 samples per pixel)
        ome_metadata = {"axes": "YXC"}
        if "PhysicalSizeX" in meta_data and "PhysicalSizeY" in meta_data:
            ome_metadata.update(
                {
                    "PhysicalSizeX": meta_data["PhysicalSizeX"],
                    "PhysicalSizeXUnit": meta_data["PhysicalSizeUnitX"],
                    "PhysicalSizeY": meta_data["PhysicalSizeY"],
                    "PhysicalSizeYUnit": meta_data["PhysicalSizeUnitY"],
                }
            )

        task = progress.add_task(
            description=f"[green]Converting {czi_path.name} to ome-tiff...",
            total=(len(subresolution_levels) + 1) * n_tiles,
        )
        sub_task = progress.add_task(
            description="[cyan]Writing resolution level 0...", total=n_tiles
        )

        with TiffWriter(ometiff_path, bigtiff=True, ome=True) as tif:
            options = dict(
                photometric="rgb",
                compression="jpeg",
                resolutionunit="MICROMETER",
                maxworkers=os.cpu_count(),
            )

            base_res = (psy.to(res_unit).magnitude, psx.to(res_unit).magnitude)

            # Highest resolution
            tif.write(
                data=load_image(
                    czi_img, (x_coords, y_coords), 0, ots, progress, (task, sub_task)
                ),
                tile=ots,
                shape=(h, w, 3),
                subifds=len(subresolution_levels),
                dtype=np.uint8,
                resolution=base_res,
                metadata=ome_metadata,
                **options,
            )
            progress.update(task, advance=1)

            # Pyramid levels
            for level in subresolution_levels:
                progress.reset(
                    sub_task,
                    total=n_tiles,
                    description=f"[cyan]Writing resolution level {level}...",
                )
                mag = 2 ** level
                ts = (round(th / mag), round(tw / mag))

                level_res = (
                    (psy / mag).to(res_unit).magnitude,
                    (psx / mag).to(res_unit).magnitude,
                )

                tif.write(
                    data=load_image(
                        czi_img, (x_coords, y_coords), level, ots, progress, (task, sub_task)
                    ),
                    tile=ts,
                    subfiletype=1,
                    shape=(round(h / mag), round(w / mag), 3),
                    resolution=level_res,
                    dtype=np.uint8,
                    metadata=None,
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


def extract_channel_names_from_czi(czi_image: CziReader, n_ch: int) -> List[str]:
    """
    Extract per channel names from the CZI raw XML metadata.

    Uses several XPath patterns in order of preference and falls back to
    generic 'Channel N' labels if no meaningful names are found.
    """
    try:
        root = ET.fromstring(czi_image.raw_metadata)
    except Exception:
        return [f"Channel {i + 1}" for i in range(n_ch)]

    # Try several plausible locations, most specific first
    xpaths = [
        ".//{*}DisplaySetting/{*}Channels/{*}Channel",
        ".//{*}Information/{*}Image/{*}Dimensions/{*}Channels/{*}Channel",
        ".//{*}Experiment/{*}ExperimentBlocks/{*}AcquisitionBlock/{*}Channels/{*}Channel",
        ".//{*}Channels/{*}Channel",
    ]

    channels = []
    for xp in xpaths:
        try:
            channels = root.findall(xp)
        except Exception:
            channels = []
        if channels:
            break

    names: List[str] = []
    for idx in range(n_ch):
        name = ""
        if idx < len(channels):
            ch_elem = channels[idx]
            for attr in ("ShortName", "Name", "Id", "DyeName", "Fluor"):
                val = ch_elem.get(attr)
                if val:
                    name = val
                    break
        if not name:
            name = f"Channel {idx + 1}"
        names.append(name)

    return names
