from pathlib import Path
from typing import Optional, Dict
from rich.progress import Progress
from tifffile import TiffWriter
import numpy as np
import os

def save_tif(image: Dict[int, np.ndarray], path: Path, metadata: Optional[Dict] = None) -> None:
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
    metadata : Optional[Dict]
        Optional JSON-serializable dict stored in ImageDescription for the *base* IFD.
        (Do NOT pass Slidekick's Metadata object here.)
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
    for lvl in levels:
        arr = image[lvl]
        if arr is None:
            raise ValueError(f"save_tif: image[{lvl}] is None.")
        arr = np.asarray(arr, order="C")  # ensure contiguous, keep dtype
        if arr.ndim == 2:
            pass  # grayscale OK
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            pass  # RGB/RGBA OK
        else:
            raise ValueError(f"save_tif: unsupported shape at level {lvl}: {arr.shape}")
        arrays.append(arr)

    # Validate channels & dtype across levels
    nchan = [(a.shape[2] if a.ndim == 3 else 1) for a in arrays]
    if any(c != nchan[0] for c in nchan):
        raise ValueError(f"save_tif: channel count mismatch across levels: {nchan}")
    C = nchan[0]
    dtype = arrays[0].dtype
    if any(a.dtype != dtype for a in arrays):
        raise ValueError("save_tif: dtype differs across levels.")

    # Photometric & compression
    photometric = "minisblack" if C == 1 else "rgb"
    # JPEG for uint8 (fast & compact), else zlib (safer for uint16 etc.)
    compression = "jpeg" if dtype == np.uint8 and C in (1, 3) else "zlib"
    tile = (1024, 1024)  # Slidekick default

    n_subifds = max(len(arrays) - 1, 0)

    # ---- Write pyramid: base IFD + SubIFDs
    with Progress() as progress:
        task = progress.add_task(f"[green]Saving image to {path.name}...", total=(n_subifds + 1))
        sub = progress.add_task("[cyan]Writing resolution level 0...", total=1)

        with TiffWriter(str(path), bigtiff=True) as tif:
            # Base (reserves space for subIFDs)
            tif.write(
                arrays[0],
                tile=tile,
                compression=compression,
                photometric=photometric,
                planarconfig="contig",
                subifds=n_subifds,
                metadata=(metadata if isinstance(metadata, dict) else None),
                maxworkers=os.cpu_count(),
            )
            progress.advance(sub, 1)
            progress.advance(task, 1)

            # SubIFDs (reduced-resolution images)
            for i, arr in enumerate(arrays[1:], start=1):
                progress.reset(sub, total=1, description=f"[cyan]Writing resolution level {i}...")
                tif.write(
                    arr,
                    tile=tile,
                    compression=compression,
                    photometric=photometric,
                    planarconfig="contig",
                    subfiletype=1,      # reduced-resolution
                    metadata=None,      # no per-subIFD JSON
                    maxworkers=os.cpu_count(),
                )
                progress.advance(sub, 1)
                progress.advance(task, 1)

        progress.remove_task(sub)

    print(f"Image saved to {path}")
