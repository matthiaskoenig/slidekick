from typing import List
import datetime
import uuid
from pathlib import Path

from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH

from slidekick.processing.baseoperator import BaseOperator

# VALIS imports
from valis import registration


class ValisRegistrator(BaseOperator):
    """
    Registers slides using VALIS. For now: Only highest resolution level is supported.
    """

    def __init__(self, metadata: List[Metadata],
                 save_img: bool = True,
                 imgs_ordered: bool = False,
                 max_processed_image_dim_px: int = 850):
        """
        channel_selection is ignored for registration (we register whole images),
        but we keep the argument signature compatible with BaseOperator usage.
        """
        self.save_img = save_img
        channel_selection = None
        self.imgs_ordered = imgs_ordered
        self.max_processed_image_dim_px = max_processed_image_dim_px
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

        # Build a list of input slide file paths from metadata.
        # Prefer path_original if present, otherwise path_storage.
        img_paths = []
        for m in self.metadata:
            # Metadata.path_storage are Path-like; cast to str for VALIS
            img_paths.append(str(m.path_storage))

        # Create a results folder for this registration run (under OUTPUT_PATH).
        # Using a timestamp + short uuid to keep results separate if apply() called multiple times.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        new_uid = f"{timestamp}-{short_id}"

        results_dir = Path(OUTPUT_PATH) / f"valis_registration_{timestamp}_{short_id}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # registered slides destination folder
        registered_slide_dst_dir = results_dir / "registered_slides"
        registered_slide_dst_dir.mkdir(parents=True, exist_ok=True)

        # Store results_dir for later access
        self.results_dir = results_dir
        self.registered_slide_dir = registered_slide_dst_dir

        # VALIS parameter:
        # - max_processed_image_dim_px controls the maximum dimension of the images used
        #   for the 'processed' image (used to find transforms). By setting it very large,
        #   VALIS will effectively use the highest-resolution level for transform estimation.

        # Initialize VALIS registrar with the list of slide image paths.
        # imgs_ordered=False lets VALIS reorder the images by similarity (unordered structure).
        # reference_img_f=None leaves selection of reference image to VALIS (it will pick center).
        #TODO: Currently hardcoded path to folder, needs adaptive copying of files into temp folder with uid or smth

        from slidekick import DATA_PATH

        registrar = registration.Valis(
            src_dir=str(DATA_PATH / "reg" / "debug"),
            dst_dir=str(results_dir),
            img_list=img_paths,
            max_processed_image_dim_px=self.max_processed_image_dim_px,
            imgs_ordered=self.imgs_ordered,
            crop="reference"
        )

        # Run registration: returns rigid and non-rigid registrar objects and an error dataframe
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()

        console.print(f"VALIS registration completed. Results directory: {results_dir}")

        input_files_used = registrar.original_img_list  # list[str]
        console.print(f"VALIS used these input files (in order it uses them): {input_files_used}")

        # Warp and save full-resolution slides to the registered_slide_dst_dir.
        # The VALIS API provides a method to warp & save the slides in native resolution.
        # We call that here so the full-resolution registered slides are available on disk.
        #
        # NOTE: API method name in VALIS is `warp_and_save_slides` (per VALIS docs/examples).
        #       If you want a different cropping method, set crop="reference" or crop="overlap".
        registrar.warp_and_save_slides(str(registered_slide_dst_dir), crop="overlap")

        console.print(f"Full-resolution registered slides saved to: {registered_slide_dst_dir}")

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
                console.print(f"Metadata updated: {meta.uid} -> {meta.path_storage.name}")
            else:
                console.print(f"[WARN] No registered file found for VALIS name '{valis_name}' (meta uid {meta.uid})")

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

    image_paths = [DATA_PATH / "reg" / "HE.tiff",
                   DATA_PATH / "reg" / "Arginase1.tiff",
                   DATA_PATH / "reg" / "KI67.tiff",
                   ]

    metadatas = [Metadata(path_original=image_path, path_storage=image_path) for image_path in image_paths]

    Registrator = ValisRegistrator(metadatas, max_processed_image_dim_px=850)

    metadatas_registered = Registrator.apply()



