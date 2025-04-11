import os
from pathlib import Path
from typing import Tuple

from slidekick.io.czi import czi2tiff


def recursive_transform(dir: Path, source_path, target_dir, **kwargs):
    for p in dir.iterdir():
        if p.is_dir():
            recursive_transform(p, source_path, target_dir, **kwargs)
        elif os.path.getsize(str(p)) / 1e6 > 20 and p.suffix == ".czi":
            transform(p, source_path, target_dir, **kwargs)


def transform(p, source_dir, target_dir, **kwargs):
    rel_path = p.relative_to(source_dir).parent
    file_name = p.stem
    target_path = target_dir.joinpath(rel_path)
    target_path.mkdir(exist_ok=True)
    target_path = target_path / f"{file_name}.ome.tiff"

    czi2tiff(p, target_path, **kwargs)


if __name__ == '__main__':
    czi_dir = Path(r"D:\data\FatLiver\HEStaining")
    target_dir = Path(r"D:\data\FatLiver\HEStaining\tiff")

    target_dir.mkdir(exist_ok=True)

    trafo_args = dict(
        tile_size=(2048, 2048),
        subresolution_levels=[1, 2, 4],
        res_unit="µm"
    )

    recursive_transform(czi_dir, czi_dir, target_dir, **trafo_args)

