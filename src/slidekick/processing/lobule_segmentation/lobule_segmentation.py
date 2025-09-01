import numpy as np
import os
from pathlib import Path
from rich.prompt import Confirm, Prompt
import zarr
from typing import List, Union, Optional
import matplotlib.pyplot as plt
from PIL import Image

from slidekick.processing.baseoperator import BaseOperator
from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH

class LobuleSegmentor(BaseOperator):
    """
    Based on https://github.com/matthiaskoenig/zonation-image-analysis/blob/develop/src/zia/pipeline/pipeline_components/segementation_component.py
    """

    def __init__(self,
                 metadata: Union[Metadata, List[Metadata]],
                 channel_selection: Union[int, List[int]] = None,
                 throw_out_ratio: float = None,
                 preview: bool = True,
                 confirm: bool = True):
        """
        @param metadata: List or single metadata object to load and use for lobule segmentation
        @param channel_selection: Number of Metadata objects that should be inverted. If None, invert none.
        Images should be inverted so that high absorbance, i.e., bright values are pericentral zones.
        Different to zia where simply high absorbance = bright
        @param throw_out_ratio: the min percentage of non_background pixel for the slide to have to not be discarded
        @param preview: whether to preview the segmentation
        @param confirm: whether to confirm the segmentation
        """
        self.throw_out_ratio = throw_out_ratio
        self.preview = preview
        self.confirm = confirm
        # make sure channel is list for later iteration
        if isinstance(channel_selection, int):
            channel_selection = [channel_selection]
        super().__init__(metadata, channel_selection)

    def _downsample_to_max_side(img: np.ndarray, max_side: int = 2048) -> np.ndarray:
        """
        Downsample image so max(height, width) == max_side, preserving aspect ratio.
        Uses Pillow LANCZOS for high-quality downscale.
        No-op if the image is already smaller than max_side.
        """
        if img is None:
            return img

        # Ensure we operate on HxW or HxWxC ndarray
        if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
            raise ValueError("Preview expects an ndarray image of shape HxW or HxWxC.")

        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return img  # already small enough

        scale = max_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Convert to PIL, resize, back to numpy
        # Normalize dtype to uint8 for display if needed
        arr = img
        if arr.dtype != np.uint8:
            # clip to [0,255] then cast for stable visualization
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        pil = Image.fromarray(arr)
        pil = pil.resize((new_w, new_h), resample=Image.LANCZOS)
        out = np.asarray(pil)

        return out

    # === add this method ===
    def _preview_images(self, images: List[np.ndarray], titles: Optional[List[str]] = None) -> None:
        """
        Downsample and render a grid preview. Max 2048 px on the longest side per image.
        """
        if not images:
            console.print("No images to preview.", style="warning")
            return

        thumbs = [self._downsample_to_max_side(im, 2048) for im in images]

        # Basic layout: up to 3 columns. Compute rows to fit all images.
        n = len(thumbs)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))

        # Reasonable figure size based on columns/rows
        fig_w = 5 * cols
        fig_h = 5 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)

        for idx, ax in enumerate(axes.flat):
            ax.axis("off")
            if idx >= n:
                continue
            im = thumbs[idx]
            # grayscale vs RGB handling
            if im.ndim == 2:
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(im)
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=9)

        plt.show()

    def _load_and_invert_images_from_metadatas(self) -> np.ndarray:
        """
        Load each image at the coarsest pyramid level available, invert selected channels,
        harmonize shapes, and stack along the last axis -> (H, W, N).
        """

        def _read_highest_level_from_store(p: Path):
            # Try Zarr/OME-NGFF multiscales first
            try:
                root = zarr.open(str(p), mode="r")
            except Exception:
                return None

            # If it's a single array, just read it
            if isinstance(root, zarr.core.Array):
                return root[...]

            # Otherwise walk groups and collect arrays; pick the smallest area (highest level index)
            candidates = []

            def _collect(g):
                for _, a in g.arrays():
                    if a.ndim >= 2:
                        candidates.append(a)
                for _, sg in g.groups():
                    _collect(sg)

            try:
                _collect(root)
            except Exception:
                candidates = []

            if candidates:
                a = min(candidates, key=lambda arr: int(arr.shape[0]) * int(arr.shape[1]))
                return a[...]
            return None

        arrays, shapes = [], []

        for i, md in enumerate(self.metadata):
            # resolve path
            p = getattr(md, "path_storage", None) or getattr(md, "path_original", None)
            p = Path(p) if p is not None else None

            img = None
            if p is not None and p.exists():
                img = _read_highest_level_from_store(p)

            # fallback to existing loader if Zarr path not present or failed
            if img is None:
                img = self.load_image(i)

            # ensure 2D
            img = np.asarray(img)
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                img = np.mean(img[..., :3], axis=-1).astype(img.dtype)

            # optional discard by foreground coverage
            if self.throw_out_ratio is not None:
                non_bg = np.count_nonzero(img != 255)
                ratio = non_bg / img.size
                if ratio < self.throw_out_ratio:
                    console.print(
                        f"Discarded {i} with non-background pixel ratio: {ratio:.3f}",
                        style="warning",
                    )
                    continue

            # optional inversion
            if self.channels is not None and i in self.channels:
                if np.issubdtype(img.dtype, np.integer):
                    info = np.iinfo(img.dtype)
                    img = info.max - img
                else:  # float images assumed in [0,1]
                    img = 1.0 - img

            arrays.append(img)
            shapes.append(img.shape[:2])

        if not arrays:
            raise RuntimeError("No images loaded after filtering.")

        # crop all to smallest H×W so stacking is defined
        h_min = min(s[0] for s in shapes)
        w_min = min(s[1] for s in shapes)
        arrays = [a[:h_min, :w_min] for a in arrays]

        # stack along the third dimension
        stack = np.dstack(arrays)  # (H, W, N)
        return stack


    def apply(self) -> Metadata:

        # Load images and invert based on metadata
        img_stack = self._load_and_invert_images_from_metadatas()

        # Apply filters


        # Show Preview after loading and filtering
        if self.preview:
            # Ensure we have a list of arrays to preview
            # zarr.Array or numpy array -> split into list; list stays as-is
            if isinstance(img_stack, list):
                imgs = img_stack
            elif hasattr(img_stack, "shape"):
                # Accept shapes like (N,H,W[,C]) and slice into a list
                if img_stack.ndim < 3:
                    imgs = [np.asarray(img_stack)]
                else:
                    imgs = [np.asarray(img_stack[i]) for i in range(img_stack.shape[0])]
            else:
                raise ValueError("Unsupported img_stack type for preview.")

            # Optional titles from metadata if available
            titles = []
            for i, md in enumerate(self.metadata):
                name = getattr(md, "name", None) or getattr(md, "sample_id", None) \
                       or getattr(md, "path", None)
                try:
                    name = Path(name).name  # shorten if it's a path
                except Exception:
                    pass
                titles.append(str(name) if name is not None else f"image_{i}")

            self._preview_images(imgs, titles)

        if self.confirm:
            apply = Confirm.ask("Continue with lobule segmentation?", default=False, console=console)
            if not apply:
                console.print("Aborted by user. No segmentation performed.", style="error")
                return self.metadata


if __name__ == "__main__":
    # Example usage
    from slidekick import DATA_PATH
    from slidekick.processing.valis_registration.valis_registration import ValisRegistrator
    from slidekick.processing.stain_separation.stain_separation import StainSeparator

    image_paths = [DATA_PATH / "reg" / "HE1.ome.tif",
                   DATA_PATH / "reg" / "HE2.ome.tif",
                   DATA_PATH / "reg" / "Arginase1.ome.tif",
                   DATA_PATH / "reg" / "KI67.ome.tif",
                   DATA_PATH / "reg" / "GS_CYP1A2.czi",
                   DATA_PATH / "reg" / "Ecad_CYP2E1.czi",
                   ]

    metadatas = [Metadata(path_original=image_path, path_storage=image_path) for image_path in image_paths]

    Registrator = ValisRegistrator(metadatas, max_processed_image_dim_px=600, max_non_rigid_registration_dim_px=600, confirm=False, preview=False)

    metadatas_registered = Registrator.apply()

    Separator_1 = StainSeparator(metadatas_registered[4], mode="fluorescence", confirm=False, preview=False)
    Separator_2 = StainSeparator(metadatas_registered[5], mode="fluorescence", confirm=False, preview=False)

    metadatas_sep_1 = Separator_1.apply()
    metadatas_sep_2 = Separator_2.apply()

    metadata_for_segmentation = metadatas_registered[2:4] + metadatas_sep_1 + metadatas_sep_2

    Segmentor = LobuleSegmentor(metadata_for_segmentation, 1)

    metadata_segmentation = Segmentor.apply()
