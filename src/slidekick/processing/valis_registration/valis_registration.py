from typing import List
import datetime
import uuid
from pathlib import Path
import shutil

from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH
from rich.prompt import Confirm
from slidekick.processing.baseoperator import BaseOperator

# VALIS imports
from valis import registration

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pyvips

class ValisRegistrator(BaseOperator):
    """
    Registers slides using VALIS. For now: Only highest resolution level is supported.
    """

    def __init__(self, metadata: List[Metadata],
                 save_img: bool = True,
                 imgs_ordered: bool = False,
                 max_processed_image_dim_px: int = 850,
                 confirm: bool = True,
                 preview: bool = True):
        """
        channel_selection is ignored for registration (we register whole images),
        but we keep the argument signature compatible with BaseOperator usage.
        """
        self.save_img = save_img
        channel_selection = None
        self.imgs_ordered = imgs_ordered
        self.max_processed_image_dim_px = max_processed_image_dim_px
        self.confirm = confirm  # Confirm if check is applied
        self.preview = preview  # Preview transformation
        super().__init__(metadata, channel_selection)

    def apply(self):
        """
        Run VALIS registration for the slides listed in self.metadata.

        Key behavior:
        - Loads the slide file paths from the metadata list and gives them to VALIS
          via img_list (this avoids intermediate array-handling problems and lets
          VALIS use its internal slide readers and pyramids).
        - Uses unordered registration (`imgs_ordered=False`) so VALIS can re-order
          by similarity and pick its own center reference image.
        - Forces VALIS to use the highest-resolution images for finding transforms
          by setting a very large `max_processed_image_dim_px` (tune if needed).
        - Warps/saves the full-resolution slides into a `registered_slides` folder
          under a temporary results directory (under OUTPUT_PATH).
        - DOES NOT return anything; instead stores useful results on `self` for
          downstream processing / saving / metadata-updating.
        """

        # Create a results folder for this registration run (under OUTPUT_PATH).
        # Using a timestamp + short uuid to keep results separate if apply() called multiple times.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        new_uid = f"{timestamp}-{short_id}"

        results_dir = Path(OUTPUT_PATH) / f"valis_registration_{timestamp}_{short_id}"
        results_dir.mkdir(parents=True, exist_ok=True)

        temp_img_dir = results_dir / "temp_imgs"
        temp_img_dir.mkdir(parents=True, exist_ok=True)

        # Copy every image metadata from path_storage to temp_img_dir
        for m in self.metadata:
            shutil.copy(m.path_storage, temp_img_dir)

        # registered slides destination folder
        registered_slide_dst_dir = results_dir / "registered_slides"
        registered_slide_dst_dir.mkdir(parents=True, exist_ok=True)

        # VALIS parameter:
        # - max_processed_image_dim_px controls the maximum dimension of the images used
        #   for the 'processed' image (used to find transforms). By setting it very large,
        #   VALIS will effectively use the highest-resolution level for transform estimation.

        # Initialize VALIS registrar with the list of slide image paths.
        # imgs_ordered=False lets VALIS reorder the images by similarity (unordered structure).
        # reference_img_f=None leaves selection of reference image to VALIS (it will pick center).

        registrar = registration.Valis(
            src_dir=str(temp_img_dir),
            dst_dir=str(results_dir),
            #img_list=img_paths, Unused for now
            max_processed_image_dim_px=self.max_processed_image_dim_px,
            imgs_ordered=self.imgs_ordered,
            crop="reference"
        )

        # Run registration: returns rigid and non-rigid registrar objects and an error dataframe
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()

        # Preview as a grid: rows = slides, cols = [original, rigid, non-rigid]
        if self.preview:
            try:
                slides = list(registrar.slide_dict.values())
                if not slides:
                    console.print("No slides found for preview.", style="error")
                    return self.metadata

                # name -> original path (invert VALIS' mapping)
                name_to_src = {v: k for k, v in registrar.name_dict.items()}

                def to_rgb(arr: np.ndarray) -> np.ndarray:
                    if arr is None:
                        return None
                    if arr.ndim == 2:
                        arr = np.repeat(arr[..., None], 3, axis=2)
                    if arr.shape[-1] > 3:
                        arr = arr[..., :3]
                    return arr

                def load_orig_downsample(src_fp: str, target_hw: tuple[int, int]) -> np.ndarray | None:
                    if not src_fp or not Path(src_fp).exists():
                        return None
                    Ht, Wt = target_hw
                    try:
                        im = pyvips.Image.thumbnail(src_fp, Wt)
                        if im.bands == 1:
                            im = im.bandjoin([im, im, im])
                        arr = np.frombuffer(im.write_to_memory(), dtype=np.uint8).reshape(im.height, im.width, im.bands)
                    except Exception:
                        try:
                            arr = imread(src_fp)
                        except Exception:
                            return None
                    if (arr.shape[0], arr.shape[1]) != (Ht, Wt):
                        arr = resize(arr, (Ht, Wt), order=1, anti_aliasing=True, preserve_range=True).astype(np.uint8)
                    if arr.ndim == 3 and arr.shape[-1] == 4:
                        arr = arr[..., :3]
                    return arr

                # Collect per-slide images
                per_slide = []
                for slide in slides:
                    name = slide.name

                    rigid_fp = getattr(slide, "rigid_reg_img_f", None)
                    nr_fp = getattr(slide, "non_rigid_reg_img_f", None)

                    rigid = imread(rigid_fp) if rigid_fp and Path(rigid_fp).exists() else None
                    nonrigid = imread(nr_fp) if nr_fp and Path(nr_fp).exists() else None

                    # choose canvas for original downsample
                    tgt = nonrigid if nonrigid is not None else rigid
                    if tgt is None:
                        # last resort: try to synthesize rigid from processed
                        try:
                            rigid = slide.warp_img(img=slide.processed_img, non_rigid=False, crop=True)
                            tgt = rigid
                        except Exception:
                            console.print(f"No registered image for '{name}'. Skipping column.", style="warning")
                            continue

                    # synthesize non-rigid if missing and dxdy exists
                    if nonrigid is None:
                        try:
                            nonrigid = slide.warp_img(img=slide.processed_img, non_rigid=True, crop=True)
                        except Exception:
                            nonrigid = None  # non-rigid likely disabled or failed

                    src_fp = name_to_src.get(name)
                    orig = load_orig_downsample(src_fp, (tgt.shape[0], tgt.shape[1]))

                    per_slide.append((name, to_rgb(orig), to_rgb(rigid), to_rgb(nonrigid)))

                if not per_slide:
                    console.print("Nothing to show.")
                    return self.metadata

                # Plot: 3 rows (orig, rigid, non-rigid) x N columns (slides)
                n = len(per_slide)
                fig_w = max(3.0 * n, 6.0)
                fig_h = 3.0 * 3
                fig, axes = plt.subplots(3, n, figsize=(fig_w, fig_h), constrained_layout=True)
                if n == 1:
                    axes = np.expand_dims(axes, 1)

                row_titles = ["original", "rigid", "non-rigid"]
                for j, (name, orig, rigid, nonrigid) in enumerate(per_slide):
                    axes[0, j].set_title(name, fontsize=10)
                    imgs = [orig, rigid, nonrigid]
                    for i in range(3):
                        ax = axes[i, j]
                        img = imgs[i]
                        if img is None:
                            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9)
                        else:
                            ax.imshow(img)
                        if j == 0:
                            ax.set_ylabel(row_titles[i], rotation=90, ha="right", va="center", fontsize=10, labelpad=18)
                            ax.set_xticks([]);
                            ax.set_yticks([])
                            for s in ax.spines.values():
                                s.set_visible(False)
                        else:
                            ax.axis("off")  # safe for non-label columns

                plt.show()

                if self.confirm:
                    apply = Confirm.ask("Apply full-resolution transformation and save outputs?", default=False,
                                        console=console)
                    if not apply:
                        console.print("Aborted by user. No warping performed.")
                        return self.metadata

            except Exception as e:
                console.print(f"Matplotlib preview failed ({e}). Continuing without preview.", style="error")

        console.print(f"VALIS registration completed. Results directory: {results_dir}", style="info")

        input_files_used = registrar.original_img_list  # list[str]
        console.print(f"VALIS used these input files (in order it uses them): {input_files_used}", style="info")

        # Warp and save full-resolution slides to the registered_slide_dst_dir.
        # The VALIS API provides a method to warp & save the slides in native resolution.
        # We call that here so the full-resolution registered slides are available on disk.
        #
        # NOTE: API method name in VALIS is `warp_and_save_slides` (per VALIS docs/examples).
        #       If you want a different cropping method, set crop="reference" or crop="overlap".
        registrar.warp_and_save_slides(str(registered_slide_dst_dir), crop="overlap")

        console.print(f"Full-resolution registered slides saved to: {registered_slide_dst_dir}", style="info")

        # Delete temp dir
        shutil.rmtree(temp_img_dir)

        # VALIS saved the transformed images to the new path as individual objects

        # We update each metadata storage path to the transformed path
        # Normalized list of input paths as VALIS used them
        valis_inputs = [str(Path(p).resolve()) for p in registrar.original_img_list]  # now actually used

        # mapping input_path -> valis_name (VALIS-assigned short name)
        name_map = {str(Path(k).resolve()): v for k, v in registrar.name_dict.items()}

        # List all files VALIS wrote (in registered_slide_dst_dir)
        out_files = sorted(registered_slide_dst_dir.glob("*"))

        # Update each metadata entry
        for meta in self.metadata:
            # Normalize the path the same way we used when creating img_paths
            orig_path = str(Path(meta.path_storage).resolve())

            # Get valis_name from name_map (fallback: use stem)
            if orig_path in name_map:
                valis_name = name_map[orig_path]
            else:
                # fallback: try to match by stem among valis_inputs (handles relative/abs differences)
                stem_matches = [p for p in valis_inputs if Path(p).stem == Path(orig_path).stem]
                if stem_matches:
                    valis_name = name_map.get(stem_matches[0], Path(orig_path).stem)
                else:
                    valis_name = Path(orig_path).stem

            # find the registered file (robust to different extensions)
            new_file = find_registered_file_for_valis_name(registered_slide_dst_dir, valis_name, out_files)

            if new_file:
                # update metadata in-place (Path objects)
                meta.path_storage = new_file
                meta.path_original = new_file
                meta.image_type = "Registered WSI (VALIS)"
                meta.uid = f"{new_uid}-{valis_name}"
                console.print(f"Metadata updated: {meta.uid} -> {meta.path_storage.name}", style="info")
            else:
                console.print(f"No registered file found for VALIS name '{valis_name}' (meta uid {meta.uid})", style="error")

        return self.metadata


# Helper: find file(s) that start with valis_name; prioritize common ome-tiff extensions
def find_registered_file_for_valis_name(registered_slide_dst_dir, valis_name, files):
    # try explicit expected extensions first
    for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        candidate = registered_slide_dst_dir / f"{valis_name}{ext}"
        if candidate.exists():
            return candidate
    # otherwise return first file whose name starts with valis_name
    prefix_matches = [f for f in files if f.name.startswith(valis_name)]
    return prefix_matches[0] if prefix_matches else None


if __name__ == "__main__":
    # Example usage
    from slidekick import DATA_PATH

    image_paths = [DATA_PATH / "reg" / "HE.ome.tif",
                   #DATA_PATH / "reg" / "Arginase1.ome.tif",
                   DATA_PATH / "reg" / "KI67.ome.tif",
                   ]

    metadatas = [Metadata(path_original=image_path, path_storage=image_path) for image_path in image_paths]

    Registrator = ValisRegistrator(metadatas, max_processed_image_dim_px=2048)

    metadatas_registered = Registrator.apply()
