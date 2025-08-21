from typing import List, Dict
import numpy as np
from skimage.morphology import remove_small_objects

from slidekick.io.metadata import Metadata
from slidekick.io import save_tif
from slidekick.console import console
from slidekick import OUTPUT_PATH

# Local helpers
from slidekick.processing.roi.roi_utils import (
    ensure_grayscale_uint8,
    detect_tissue_mask,
    largest_bbox
)

from slidekick.processing.baseoperator import BaseOperator

# Operator class performing detection and extraction in a single step
class RoiDetector(BaseOperator):
    """Operator that detects tissue ROIs and extracts the cropped image.

    Constructor keyword arguments
    -----------------------------
    morphological_radius : int
        Radius of the closing structuring element (default 5).
    min_area : int
        Minimum area (in pixels) for connected components to be considered
        (default 5000). Small components are removed using this threshold.
    save_img : bool
        If True, save the cropped image (default True).
    """

    def __init__(self,
                 metadata: Metadata,
                 channel_selection: List[int],
                 morphological_radius: int = 5,
                 min_area: int = 5000,
                 save_img: bool = True):
        self.morphological_radius = morphological_radius
        self.min_area = min_area
        self.save_img = save_img

        super().__init__(metadata, channel_selection)


    def apply(self):
        """Detect ROI, save mask and extracted ROI, update metadata.

        Returns the original image to preserve compatibility with other operators
        in the pipeline.
        """
        # Get image from stored metadata and extract image channel[s]

        # This returns a dictionary where keys are image levels (e.g., 0, 1, 2) and values are zarr arrays
        image = self.load_image()

        # Chosen channels are stored in self.channels, which is a list of tuples (image_index, channel_index)
        # For example, if self.channels = [(0, 0), (0, 1), (0, 2)], it means we want to extract channels 0, 1, and 2

        # Now we need to extract the image data for the selected channels
        image_data = {}
        for level, img in image.items():
            if img.ndim == 3 and img.shape[2] >= 3:
                # Extract the selected channels for the current image level
                selected_channels = [img[:, :, idx] for _, idx in self.channels if idx < img.shape[2]]
                if len(selected_channels) == 3:
                    image_data[level] = np.stack(selected_channels, axis=-1)
                else:
                    raise ValueError(
                        f"Not enough channels selected for image at level {level}. Expected 3, got {len(selected_channels)}.")

        # Choose the most downsampled (smallest area) level for detection (faster)
        detection_level = min(image_data.keys(), key=lambda l: image_data[l].shape[0] * image_data[l].shape[1])
        detection_img = image_data[detection_level]

        # Convert detection image to grayscale and compute mask
        gray = ensure_grayscale_uint8(detection_img)
        mask = detect_tissue_mask(gray, self.morphological_radius)

        # Remove small spurious objects
        mask = remove_small_objects(mask, min_size=self.min_area)

        # Find bounding box in detection level coords
        bbox = largest_bbox(mask)
        if bbox is None:
            console.print("No ROI detected; skipping crop and metadata save.", style="warning")
            return None, None

        # Map bbox from detection level -> full resolution (prefer level 0)
        if isinstance(image, dict) or hasattr(image, 'items'):
            # prefer raw full resolution under key 0 if present, otherwise pick the largest level
            raw_full = image.get(0) or image.get(
                max(image_data.keys(), key=lambda l: image_data[l].shape[0] * image_data[l].shape[1]))
        else:
            raw_full = image

        det_h, det_w = detection_img.shape[:2]
        full_h, full_w = raw_full.shape[:2]
        scale_x = full_w / float(det_w)
        scale_y = full_h / float(det_h)

        x_det, y_det, w_det, h_det = bbox
        x_full = int(round(x_det * scale_x))
        y_full = int(round(y_det * scale_y))
        w_full = int(round(w_det * scale_x))
        h_full = int(round(h_det * scale_y))
        roi_bounds_full = (x_full, y_full, w_full, h_full)

        roi_img = {}

        # Determine reference full-resolution image in image_data for scaling calculations.
        # Prefer level 0 if present; otherwise choose the level with largest pixel area.
        if 0 in image_data:
            ref_level = 0
        else:
            ref_level = max(image_data.keys(), key=lambda L: image_data[L].shape[0] * image_data[L].shape[1])

        ref_img = image_data[ref_level]
        ref_h, ref_w = ref_img.shape[:2]

        x_full, y_full, w_full, h_full = roi_bounds_full

        # Crop each level by scaling the full-resolution bbox into the level coordinate space
        for level, lvl_img in image_data.items():
            lvl_h, lvl_w = lvl_img.shape[:2]

            # compute scale factors full -> this level
            scale_x = lvl_w / float(ref_w)
            scale_y = lvl_h / float(ref_h)

            # scale bbox into level coords (round to nearest pixel)
            x_l = int(round(x_full * scale_x))
            y_l = int(round(y_full * scale_y))
            w_l = int(round(w_full * scale_x))
            h_l = int(round(h_full * scale_y))

            # clip to level bounds
            y0 = max(0, y_l)
            x0 = max(0, x_l)
            y1 = min(lvl_h, y_l + h_l)
            x1 = min(lvl_w, x_l + w_l)

            # take crop (works for 2D or 3D arrays)
            roi_img[int(level)] = lvl_img[y0:y1, x0:x1].copy()

        # Save and update metadata
        roi_metadata = self.save_and_update_metadata(roi_img)

        return roi_img, roi_metadata


    def save_and_update_metadata(self, cropped_data) -> Metadata:
        """
        Update the metadata with the cropped image data.
        Then save the metadata to the storage path.
        This method is called after the roi is detected
        Note: This creates a new metadata object with the cropped image data based on the original metadata, but a
        new path for the cropped image, other description, and uid
        """
        if isinstance(self.metadata, list):
            s_metadata = self.metadata[0]
        else:
            s_metadata = self.metadata

        if self.save_img:
            storage_path = s_metadata.path_storage.with_name(s_metadata.path_storage.stem + "_cropped.tiff")
        else:
            storage_path = None

        # Create a new metadata object for the cropped image
        cropped_metadata = Metadata(
            path_original=s_metadata.path_original,
            path_storage=storage_path,
            image_type=s_metadata.image_type,
            #stains={idx: f"{name}_cropped" for idx, _, name in s_metadata.channels},
            uid=f"{s_metadata.uid}_cropped"
        )

        # Save the cropped metadata
        cropped_metadata.save(output_path=OUTPUT_PATH)  #
        console.print(f"Cropped metadata saved to {cropped_metadata.path_storage}")

        # Save the new image data to the storage path
        if self.save_img:
            save_tif(cropped_data, cropped_metadata.path_storage, metadata=cropped_metadata)

        return cropped_metadata

if __name__ == "__main__":
    # Example usage
    from slidekick import DATA_PATH

    image_path = DATA_PATH / "'APAP 500mg m4 HE'.tiff"
    metadata = Metadata(path_original=image_path, path_storage=image_path)

    # Initialize the ROI detector with the metadata and channel selection
    detector = RoiDetector(metadata, channel_selection=[0, 1, 2])

    # Apply the stain normalization
    metadata_updated = detector.apply()
