from abc import ABC
from typing import List, Any, Tuple, Union, Optional
from slidekick.io.metadata import Metadata
from slidekick.io import read_wsi
from slidekick.console import console
import zarr



class BaseOperator(ABC):
    """
    Abstract base class for all operators. We include three main types of operations:
    1. Append a channel to an image.
    2. Merge channels from one or more images into a new image.
    3. Modify one or more channels in an image.

    This class provides a common interface and utility methods for channel selection
    and metadata handling.

    Additional functions include the saving of the resulting image and metadata.

    Subclasses must implement the `apply` method to define the specific operation.
    """

    def __init__(
        self,
        metadata: Union[Metadata, List[Metadata]],
        channel_selection: Union[Tuple[int, int], List[Tuple[int, int]], List[int]]
    ) -> None:
        self.metadata = metadata if isinstance(metadata, list) else [metadata]

        # If channel_selection is a List of integers, convert it to a list of tuples with the first image index as 0
        if isinstance(channel_selection, list) and all(isinstance(c, int) for c in channel_selection):
            channel_selection = [(0, c) for c in channel_selection]
        # If channel_selection is a single tuple, convert it to a list of one tuple
        if isinstance(channel_selection, tuple) and len(channel_selection) == 2:
            channel_selection = [channel_selection]

        self.channels = channel_selection

        # TODO: Validate channel_selection to ensure it contains valid indices
        console.print(f"Initialized operator with channel references: {self.channels}", style="info")

    def load_image(self, metadata_idx: int = 0) -> dict[int, zarr.Array]:
        """
        Load the image data for the specified index from the metadata.
        """
        if metadata_idx is None:
            metadata_idx = 0

        if metadata_idx >= len(self.metadata):
            raise IndexError(f"Metadata index {metadata_idx} out of range for {len(self.metadata)} metadata objects.")
        metadata = self.metadata[metadata_idx]
        if not isinstance(metadata, Metadata):
            raise TypeError("Metadata must be an instance of Metadata class.")

        img, _ = read_wsi(metadata.path_storage)

        return img


    def extract_channels(self, image: dict[int, zarr.Array]) -> List[zarr.Array]:
        """
        Extract the channels from the image based on the selected channels.
        Returns a list of zarr arrays for the selected channels.
        """
        extracted_channels = []
        for img_idx, channel_idx in self.channels:
            if img_idx < len(self.metadata):
                metadata = self.metadata[img_idx]
                if channel_idx < len(image):
                    extracted_channels.append(image[channel_idx])
                else:
                    console.print(f"Channel index {channel_idx} out of range for image {img_idx}.", style="error")
            else:
                console.print(f"Image index {img_idx} out of range for metadata.", style="error")
        return extracted_channels
