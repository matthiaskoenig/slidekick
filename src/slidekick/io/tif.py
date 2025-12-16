from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from rich.progress import Progress
from tifffile import TiffWriter
import numpy as np
import os


def _normalize_ome_length_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    u = str(unit).strip()
    if not u:
        return None
    if u.lower() in {"um", "micron", "microns", "micrometer", "micrometers"}:
        return "µm"
    if u in {"μm", "µm"}:
        return "µm"
    return u


def _extract_physical_pixel_size(meta: Any) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Extract (x,y,unit) from Slidekick Metadata object or dict-like metadata."""
    if meta is None:
        return None, None, None

    # Slidekick Metadata object
    if hasattr(meta, "get_physical_pixel_size"):
        try:
            x, y, unit = meta.get_physical_pixel_size()
            return x, y, _normalize_ome_length_unit(unit)
        except Exception:
            pass

    # Dict-like
    if isinstance(meta, dict):
        res = meta.get("resolution")
        unit = meta.get("resolution_unit")
        x = y = None
        if isinstance(res, dict):
            try:
                x = float(res.get("x")) if res.get("x") is not None else None
            except Exception:
                x = None
            try:
                y = float(res.get("y")) if res.get("y") is not None else None
            except Exception:
                y = None
        return x, y, _normalize_ome_length_unit(unit)

    return None, None, None


def _inject_physical_size_into_ome_metadata(ome_meta: Dict, meta_source: Any) -> Dict:
    """Add PhysicalSizeX/Y (+ unit) to ome_meta if missing. Returns a shallow-copied dict."""
    if not isinstance(ome_meta, dict):
        return ome_meta

    # Do not override if caller already provided calibration explicitly
    if "PhysicalSizeX" in ome_meta or "PhysicalSizeY" in ome_meta:
        return ome_meta

    x, y, unit = _extract_physical_pixel_size(meta_source)
    if x is None and y is None:
        return ome_meta

    out = dict(ome_meta)  # shallow copy
    if x is not None:
        out["PhysicalSizeX"] = float(x)
        if unit is not None:
            out["PhysicalSizeXUnit"] = unit
    if y is not None:
        out["PhysicalSizeY"] = float(y)
        if unit is not None:
            out["PhysicalSizeYUnit"] = unit

    return out


def save_tif(
    image: Dict[int, np.ndarray],
    path: Path,
    metadata: Optional[Any] = None,
    ome_metadata: Optional[Dict] = None,
) -> None:
    """
    Save a multi-resolution pyramid to a tiled, pyramidal TIFF.

    Parameters
    ----------
    image : Dict[int, np.ndarray]
        Mapping: level index -> image array (0 = full res). Each value can be:
        - HxW (grayscale) or HxWx3 (RGB) or HxWx4 (RGBA).
        All levels must share the same channel count & dtype.
    path : Path
        Output file. '.tiff' will be enforced.
    metadata : Optional[Any]
        Optional metadata. If this is a dict, it may be stored in ImageDescription (when ome_metadata is None).
        If this is a Slidekick Metadata object and ome_metadata is provided, physical pixel size calibration
        will be injected into OME metadata.
    ome_metadata : Optional[Dict]
        Optional OME metadata dict. If provided, the file is written as OME-TIFF
        (using tifffile's `ome=True`) and this dict is passed to tifffile.
    """
    # ---- Prep path & inputs
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".tiff":
        path = path.with_suffix(".tiff")

    if not image:
        raise ValueError("save_tif: empty image dict.")

    # Normalize levels: 0 (full res), 1, 2, ...
    levels = sorted(int(k) for k in image.keys())
    arrays = []
    layout = None  # "YX" (2D), "YXS" (RGB/RGBA), or "CYX" (channel-first stack)
    for lvl in levels:
        arr = image[lvl]
        if arr is None:
            raise ValueError(f"save_tif: image[{lvl}] is None.")
        arr = np.asarray(arr, order="C")  # ensure contiguous, keep dtype

        if arr.ndim == 2:
            cur_layout = "YX"
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            cur_layout = "YXS"  # samples-last RGB/RGBA
        elif arr.ndim == 3:
            cur_layout = "CYX"  # channel-first (C, Y, X), used for OME channel stacks (including C=1)
        else:
            raise ValueError(f"save_tif: unsupported shape at level {lvl}: {arr.shape}")

        if layout is None:
            layout = cur_layout
        elif cur_layout != layout:
            raise ValueError(f"save_tif: layout mismatch across levels: {layout} vs {cur_layout}")

        arrays.append(arr)

    # Validate channels & dtype across levels
    if layout == "YXS":
        nchan = [a.shape[2] for a in arrays]
    elif layout == "CYX":
        nchan = [a.shape[0] for a in arrays]
    else:
        nchan = [1 for _ in arrays]

    if any(c != nchan[0] for c in nchan):
        raise ValueError(f"save_tif: channel count mismatch across levels: {nchan}")

    C = int(nchan[0])
    dtype = arrays[0].dtype
    if any(a.dtype != dtype for a in arrays):
        raise ValueError("save_tif: dtype differs across levels.")

    # Photometric & compression
    is_rgb = (layout == "YXS")
    photometric = "rgb" if is_rgb else "minisblack"

    # Be conservative: keep fluorescence/channel stacks lossless.
    # JPEG only for true RGB uint8.
    compression = "jpeg" if (is_rgb and dtype == np.uint8 and C in (3, 4)) else "zlib"

    tile = (1024, 1024)  # Slidekick default

    n_subifds = max(len(arrays) - 1, 0)

    # Decide which metadata to pass to tifffile
    if ome_metadata is not None:
        # Ensure OME-XML contains PhysicalSizeX/Y so QuPath shows µm/px.
        ome_metadata = _inject_physical_size_into_ome_metadata(ome_metadata, metadata)
        base_metadata = ome_metadata
        ome_flag = True
    else:
        base_metadata = metadata if isinstance(metadata, dict) else None
        ome_flag = False

    # ---- Write pyramid: base IFD + SubIFDs
    with Progress() as progress:
        task = progress.add_task(f"[green]Saving image to {path.name}...", total=(n_subifds + 1))
        sub = progress.add_task("[cyan]Writing resolution level 0...", total=1)

        # If ome_metadata is provided, enable OME mode.
        with TiffWriter(str(path), bigtiff=True, ome=ome_flag) as tif:
            # Base (reserves space for subIFDs)
            base_kwargs = dict(
                tile=tile,
                compression=compression,
                photometric=photometric,
                subifds=n_subifds,
                metadata=base_metadata,
                maxworkers=os.cpu_count(),
            )
            if layout == "YXS":
                base_kwargs["planarconfig"] = "contig"

            tif.write(arrays[0], **base_kwargs)

            progress.advance(sub, 1)
            progress.advance(task, 1)

            # SubIFDs (reduced-resolution images)
            for i, arr in enumerate(arrays[1:], start=1):
                progress.reset(sub, total=1, description=f"[cyan]Writing resolution level {i}...")
                sub_kwargs = dict(
                    tile=tile,
                    compression=compression,
                    photometric=photometric,
                    subfiletype=1,  # reduced-resolution
                    metadata=None,  # no per-subIFD JSON/OME metadata
                    maxworkers=os.cpu_count(),
                )
                if layout == "YXS":
                    sub_kwargs["planarconfig"] = "contig"

                tif.write(arr, **sub_kwargs)

                progress.advance(sub, 1)
                progress.advance(task, 1)

        progress.remove_task(sub)

    print(f"Image saved to {path}")
