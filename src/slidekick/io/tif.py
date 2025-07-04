from pathlib import Path
from typing import Optional, Dict
from rich.progress import Progress
from tifffile import TiffWriter
import os

def save_tif(image: Dict, path: Path, metadata: Optional[Dict]) -> None:
    """
    Save a numpy array as a TIFF file.

    Args:
        image_data: The image data to save.
        path (Path): The path where the TIFF file will be saved.
        metadata (Optional[Dict]): Optional metadata to include in the TIFF file.
    """

    # TODO: Needs proper metadata handling (see czi2tiff for example, unify this with that function)

    # Ensure the directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    # Check if the path has a .tiff suffix, if not, add it
    if not path.suffix.lower() == '.tiff':
        path = path.with_suffix('.tiff')

    n_tiles = 1
    subresolution_levels = 1 # Debug

    with Progress() as progress:
        task = progress.add_task(description=f"[green]Saving image to {path.name}...", total=(len(subresolution_levels) + 1) * n_tiles)

        sub_task = progress.add_task(description="[cyan]Writing resolution level 0...", total=n_tiles)

        with TiffWriter(str(path), bigtiff=True) as tif:

            options = dict(
                photometric='rgb',
                compression='jpeg',
                resolutionunit='MICROMETER',
                maxworkers=os.cpu_count())

            tif.write(image, **options)

            progress.update(task, advance=1)

            for level in subresolution_levels:
                progress.reset(sub_task, total=n_tiles, description=f"[cyan]Writing resolution level {level}...")

                break

                progress.update(task, advance=1)

            progress.remove_task(sub_task)

    print(f"Image saved to {path}")