from ..baseoperator import BaseOperator
import numpy as np
from .stain_utils import normalize_stain
from slidekick.console import console
from slidekick.metadata.metadata import Metadata
from typing import Union, List, Tuple

class StainNormalizer(BaseOperator):
    """
    Stain normalization operator that applies a stain normalization algorithm to the selected channels
    of the provided whole-slide images (WSIs).
    """

    def init(self, metadata: Metadata,
             channel_selection: List[int],
             HERef: np.ndarray = np.array([[0.65, 0.2159], [0.70, 0.8012], [0.29, 0.5581]]),
             maxCRef: np.ndarray = np.array([1.9705, 1.0308])) -> None:
        """
        Initialize the stain normalizer operator
        """
        # Ensure that 3 channels and 1 image, i.e., metadata object are selected before init call of the superclass
        if not isinstance(metadata, Metadata):
            raise ValueError("Metadata must be an instance of Metadata class.")
        if not isinstance(channel_selection, list) or len(channel_selection) != 3:
            raise ValueError("Channel selection must be a list of 3 channel indices.")

        self.HERef = HERef
        self.maxCRef = maxCRef

        # Call the init method of the parent class to set up metadata and channel selection
        super().__init__(metadata, channel_selection)


    def apply(self):
        """
        Apply the stain normalization to the selected channels of the images.
        This method is called by the BaseOperator's run method.
        """
        # Get image from stored metadata and extract image channel[s]

        # This returns a dictionary where keys are image levels (e.g., 0, 1, 2) and values are zarr arrays
        image = self.metadata.load_image()

        # Chosen channels are stored in self.channels, which is a list of tuples (image_index, channel_index, channel_name)

        # Now we need to extract the image data for the selected channels
        image_data = {}
        for level, img in image.items():
            if img.ndim == 3 and img.shape[2] >= 3:
                # Extract the selected channels for the current image level
                selected_channels = [img[:, :, idx] for _, idx, _ in self.channels if idx < img.shape[2]]
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


        # Update the metadata with the normalized image data and save it
        # TODO
