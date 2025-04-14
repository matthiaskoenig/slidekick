from dataclasses import dataclass, field
from typing import Dict, Optional
import uuid
from datetime import datetime
from pathlib import Path

@dataclass
class Metadata:
    """Information for whole slide scans.

    Information on:
    - unique identifier,
    - what is stained in which channel
    - raw image format: czi, ndpi, qptiff
    - image type: fluorescence, immunohistochemistry, brightfield
    - magnification # TODO
    - resolution # TODO
    - resolution unit # TODO
    """

    storage: Path
    filename: str
    raw_format: str
    magnification: float = None
    resolution: Dict[float, float] = field(default_factory=dict)
    resolution_unit: str = None
    uid: str = field(init=False)
    image_type: Optional[str] = None
    stains: Dict[int, str] = field(default_factory=dict)
    annotations: Dict = field(default_factory=dict)

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        self.uid = f"{timestamp}_{short_id}"

    def set_stains(self, stains: Dict[int, str]) -> None:
        self.stains = stains

    def set_image_type(self, image_type: str) -> None:
        self.image_type = image_type

    def set_annotations(self, annotations: Dict) -> None:
        self.annotations = annotations
