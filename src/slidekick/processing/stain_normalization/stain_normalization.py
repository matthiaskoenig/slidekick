from slidekick.processing.baseoperator import BaseOperator
import numpy as np
from slidekick.processing.stain_normalization.stain_utils import normalize_stain
from slidekick.console import console
from slidekick.io.metadata import Metadata
from typing import List
from slidekick.io import save_tif

class StainNormalizer(BaseOperator):
    """
    Stain normalization operator that applies a stain normalization algorithm to the selected channels
    of the provided whole-slide images (WSIs).
    """

    def __init__(
            self,
            metadata: Metadata,
            channel_selection: List[int],
            HERef: np.ndarray = np.array([[0.65, 0.2159], [0.70, 0.8012], [0.29, 0.5581]]),
            maxCRef: np.ndarray = np.array([1.9705, 1.0308])) -> None:

        # Ensure that 3 channels and 1 image, i.e., metadata object are selected before init call of the superclass
        if not isinstance(metadata, Metadata):
            raise ValueError("Metadata must be an instance of Metadata class.")
        if not isinstance(channel_selection, list) or len(channel_selection) != 3:
            raise ValueError("Channel selection must be a list of 3 channel indices.")

        self.HERef = HERef
        self.maxCRef = maxCRef

        # Call the init method of the parent class to set up metadata and channel selection
        super().__init__(metadata, channel_selection)


    def save_and_update_metadata(self, normalized_data) -> Metadata:
        """
        Update the metadata with the normalized image data.
        Then save the metadata to the storage path.
        This method is called after the stain normalization is applied.
        Note: This creates a new metadata object with the normalized image data based on the original metadata, but a
        new path for the normalized image, other description, and uid
        """
        # Create a new metadata object for the normalized image
        normalized_metadata = Metadata(
            path_original=self.metadata.path_original,
            path_storage=self.metadata.path_storage.with_suffix(".normalized.tiff"),
            image_type=self.metadata.image_type,
            stains={idx: f"{name}_normalized" for idx, _, name in self.channels},  # Use the selected channels
            uid=f"{self.metadata.uid}_normalized"
        )

        # Save the normalized metadata
        normalized_metadata.save()
        console.print(f"Normalized metadata saved to {normalized_metadata.path_storage}")

        # Save the new image data to the storage path
        save_tif(normalized_data, normalized_metadata.path_storage, metadata=normalized_metadata)

        return normalized_metadata


    def apply(self):
        """
        Apply the stain normalization to the selected channels of the images.
        This method is called by the BaseOperator's run method.
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
                    raise ValueError(f"Not enough channels selected for image at level {level}. Expected 3, got {len(selected_channels)}.")

        # Normalize the stains for each image level
        normalized_data = {}
        for level, img in image_data.items():
            if img.ndim == 3 and img.shape[2] >= 3:  # Ensure we have at least 3 channels selected
                normalized_img = normalize_stain(img, self.HERef, self.maxCRef)
                normalized_data[level] = normalized_img
            else:
                raise ValueError(f"Image at level {level} does not have enough channels for stain normalization.")

        # Update the metadata with the normalized image data
        normalized_metadata = self.save_and_update_metadata(normalized_data)

        return normalized_data, normalized_metadata


if __name__ == "__main__":
    # Example usage
    from slidekick import DATA_PATH

    image_path = DATA_PATH / "'APAP 500mg m4 HE'.tiff"
    metadata = Metadata(path_original=image_path, path_storage=image_path)

    # Initialize the stain normalizer with the metadata and channel selection
    normalizer = StainNormalizer(metadata, channel_selection=[0, 1, 2])

    # Apply the stain normalization
    metadata_updated = normalizer.apply()
