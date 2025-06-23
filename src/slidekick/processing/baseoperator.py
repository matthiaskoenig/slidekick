from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union
from slidekick.metadata.metadata import Metadata
from slidekick.console import console


def ensure_list(x: Union[Any, List[Any]]) -> List[Any]:
    return x if isinstance(x, list) else [x]

def normalize_channel_selection(
    selection: Union[Tuple[int, int], List[Tuple[int, int]]],
    metadata_list: List[Metadata]
) -> List[Tuple[int, int, str]]:
    """
    Normalize a selection of (image_index, channel_index) pairs into a list of
    (image_index, channel_index, channel_name), verifying against stains.
    """
    selection = ensure_list(selection)
    normalized = []
    for item in selection:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Each channel selection must be a tuple of (image_index, channel_index), got: {item}")
        image_idx, channel_idx = item
        if image_idx >= len(metadata_list):
            raise ValueError(f"Image index {image_idx} out of range for metadata list of length {len(metadata_list)}")
        metadata = metadata_list[image_idx]
        if channel_idx not in metadata.stains:
            raise ValueError(f"Channel index {channel_idx} not in stains for image {image_idx}")
        normalized.append((image_idx, channel_idx, metadata.stains[channel_idx]))
    return normalized


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
        self.metadata = ensure_list(metadata)

        # Allow simpler input (e.g., channel index only) when there's just one metadata
        if isinstance(self.metadata, list) and len(self.metadata) == 1:
            self.metadata = self.metadata[0]
            # If channel_selection is a List of ints
            if isinstance(channel_selection, tuple) and len(channel_selection) == 1:



        if len(self.metadata) == 1 and not isinstance(channel_selection, list) and not isinstance(channel_selection, tuple):
            raise ValueError("Single channel index must be provided as a tuple (0, channel_index) or list of such tuples.")

        self.channels = normalize_channel_selection(channel_selection, self.metadata)
        console.print(f"Initialized operator with channel references: {self.channels}", style="info")
