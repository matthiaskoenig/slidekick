import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Tuple
import uuid
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import tifffile

from slidekick.console import console


@dataclass
class Metadata:
    path_original: Path
    path_storage: Path
    magnification: float = None
    filename_original: str = None
    filename_stored: str = None
    raw_format_original: str = None
    raw_format_stored: str = None
    resolution: Dict[float, float] = field(default_factory=dict)
    resolution_unit: str = None
    uid: str = field(default=None)
    image_type: Optional[str] = None
    stains: Dict[int, str] = field(default_factory=dict)
    # OME Channel Color values as signed 32-bit ARGB ints (0xAARRGGBB), one per channel index.
    channel_colors: Dict[int, int] = field(default_factory=dict)
    annotations: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.uid:
            timestamp = datetime.now().strftime("%Y%m%d")
            short_id = uuid.uuid4().hex[:8]
            self.uid = f"{timestamp}-{short_id}"

        self.filename_original = self.path_original.name
        self.filename_stored = self.path_storage.name
        self.raw_format_original = self.path_original.suffix
        self.raw_format_stored = self.path_storage.suffix

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=json_default, indent=4)

    def save(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        metadata_file = output_path / f"{self.uid}_metadata.json"
        with open(metadata_file, "w") as f:
            f.write(self.to_json())
        console.print(f"Metadata saved to {metadata_file}", style="info")

    @classmethod
    def from_json(cls, json_path: Path) -> "Metadata":
        with open(json_path, "r") as f:
            data = json.load(f)

        data["path_original"] = Path(data["path_original"])
        data["path_storage"] = Path(data["path_storage"])

        # Convert string keys back to integers in dict fields
        if "stains" in data:
            data["stains"] = {int(k): v for k, v in data["stains"].items()}

        if "channel_colors" in data:
            data["channel_colors"] = {int(k): int(v) for k, v in data["channel_colors"].items()}

        metadata = cls(**data)

        if not metadata.path_storage.exists():
            console.print(f"Missing stored image at {metadata.path_storage}", style="warning")

        return metadata

    def set_image_type(self, image_type: str) -> None:
        self.image_type = image_type

    def set_stains(self, stains: Dict[int, str]) -> None:
        self.stains = stains

    def set_annotations(self, annotations: Dict) -> None:
        self.annotations = annotations

    def set_channel_colors(self, channel_colors: Dict[int, int]) -> None:
        self.channel_colors = channel_colors

    def _ome_source_path(self) -> Optional[Path]:
        """
        Return the best path to read OME-XML from.

        If path_storage is a .czi and a sibling .tiff exists (from CZI -> TIFF conversion),
        use the .tiff for OME extraction.
        """
        p = Path(self.path_storage)
        if p.suffix.lower() == ".czi":
            t = p.with_suffix(".tiff")
            if t.exists():
                p = t

        if p.exists() and p.suffix.lower() in {".tif", ".tiff"}:
            return p
        return None

    @staticmethod
    def _ome_rgba_int(r: int, g: int, b: int, a: int = 255) -> int:
        """
        Signed 32-bit RGBA integer (0xRRGGBBAA) as used by OME Channel Color.

        Note: OME Color is RGBA (byte order), stored as a signed 32-bit integer.
        Using ARGB here will lead to channel color swaps in readers that assume RGBA.
        """
        v = ((r & 255) << 24) | ((g & 255) << 16) | ((b & 255) << 8) | (a & 255)
        if v >= 2 ** 31:
            v -= 2 ** 32
        return int(v)

    @staticmethod
    def _default_palette_rgb(i: int) -> Tuple[int, int, int]:
        # Deterministic palette with RGB as the first three channels
        palette: List[Tuple[int, int, int]] = [
            (255, 0, 0),  # red
            (0, 255, 0),  # green
            (0, 0, 255),  # blue
            (255, 0, 255),  # magenta
            (0, 255, 255),  # cyan
            (255, 255, 0),  # yellow
            (255, 128, 0),  # orange
            (153, 0, 255),  # violet
            (0, 179, 77),  # teal
            (255, 153, 153),  # pink
        ]
        return palette[i % len(palette)]

    @staticmethod
    def _guess_rgb_from_name(name: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Best-effort heuristic mapping when OME Channel Color is missing.
        This is intentionally centralized in Metadata (not stain separation).
        """
        n = (name or "").lower()

        if any(k in n for k in ("dapi", "hoechst")):
            return (0, 0, 255)
        if any(k in n for k in ("fitc", "gfp", "alexa 488", "af488", "cy2")):
            return (0, 255, 0)
        if any(k in n for k in ("tritc", "cy3", "alexa 555", "alexa 568", "af555", "af568",
                                "texas red", "tdtomato", "mcherry")):
            return (255, 0, 0)
        if any(k in n for k in ("cy5", "alexa 647", "af647", "far red", "far-red")):
            return (255, 0, 255)

        return fallback

    def enrich_from_storage(self, overwrite: bool = False) -> None:
        """
        Populate stains and channel_colors from OME-XML in the transformed TIFF, if available.

        - Does not require reading pixel data.
        - Does not override user-provided stains/colors unless overwrite=True.
        """
        src = self._ome_source_path()
        if src is None:
            return

        try:
            with tifffile.TiffFile(str(src)) as tf:
                ome_xml = tf.ome_metadata
        except Exception:
            return

        if not ome_xml:
            return

        try:
            root = ET.fromstring(ome_xml)
        except Exception:
            return

        channels = root.findall(".//{*}Pixels/{*}Channel")
        if not channels:
            return

        names: List[str] = []
        colors: List[Optional[int]] = []

        for ch in channels:
            names.append(ch.get("Name") or "")
            c = ch.get("Color")
            if c is None:
                colors.append(None)
            else:
                try:
                    colors.append(int(c))
                except Exception:
                    colors.append(None)

        # stains (names)
        if overwrite or not self.stains:
            self.stains = {i: (names[i] if names[i] else f"ch{i}") for i in range(len(names))}
        else:
            for i, nm in enumerate(names):
                if i not in self.stains or not self.stains[i]:
                    self.stains[i] = nm if nm else f"ch{i}"

        # channel_colors (OME Color if present)
        if any(c is not None for c in colors):
            if overwrite or not self.channel_colors:
                self.channel_colors = {i: int(colors[i]) for i in range(len(colors)) if colors[i] is not None}
            else:
                for i, col in enumerate(colors):
                    if col is None:
                        continue
                    if i not in self.channel_colors:
                        self.channel_colors[i] = int(col)

    def ensure_channel_metadata(self, n_channels: int) -> None:
        """
        Ensure stains and channel_colors exist for channel indices [0..n_channels-1].

        - First tries to enrich from transformed OME-TIFF metadata.
        - Fills any missing names as ch0, ch1, ...
        - Fills any missing colors from name heuristics, else deterministic palette.
        """
        if n_channels is None or n_channels < 1:
            return

        # Try to pull from OME if available (no-op if already filled)
        self.enrich_from_storage(overwrite=False)

        if self.stains is None:
            self.stains = {}
        for i in range(n_channels):
            if i not in self.stains or not self.stains[i]:
                self.stains[i] = f"ch{i}"

        if self.channel_colors is None:
            self.channel_colors = {}
        for i in range(n_channels):
            if i in self.channel_colors and self.channel_colors[i] is not None:
                continue
            fallback_rgb = self._default_palette_rgb(i)
            name_i = self.stains.get(i, "")
            r, g, b = self._guess_rgb_from_name(name_i, fallback_rgb)
            self.channel_colors[i] = self._ome_rgba_int(r, g, b)

@dataclass
class FileList:
    """List of all files as uid list metadata for the current instance."""
    files: Dict[str, Metadata] = field(default_factory=dict)

    def add_file(self, metadata: Metadata) -> None:
        """Add a file to the list."""
        uid = metadata.uid
        self.files[uid] = metadata

    def remove_file(self, uid: str) -> None:
        """Remove a file from the list."""
        if uid in self.files:
            del self.files[uid]

    def get_file(self, uid: str) -> Optional[Metadata]:
        """Get a file by its UID."""
        return self.files.get(uid)

    def save_all(self, output_path: Path) -> None:
        """Save all metadata files and the file list to a directory."""
        for uid, metadata in self.files.items():
            metadata.save(output_path)
        self.save_list(output_path)

    def save_list(self, output_path: Path) -> None:
        """Save the file list to a file."""
        file_list_file = output_path / "file_list.json"
        # Ensure the directory exists
        file_list_file.parent.mkdir(parents=True, exist_ok=True)
        # Check if the file already exists
        if file_list_file.exists():
            console.print(f"File list already exists at {file_list_file}. Overwriting.", style="warning")

        with open(file_list_file, "w") as f:
            f.write(self.to_json())
        console.print(f"File list saved to {file_list_file}", style="info")

    def to_json(self) -> str:
        return json.dumps(
            {uid: asdict(metadata) for uid, metadata in self.files.items()},
            default=json_default,
            indent=4
        )

    @classmethod
    def from_json(cls, json_path: Path) -> "FileList":
        """Load a file list from a JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        file_list = cls()
        for uid, metadata in data.items():
            metadata_json_path = json_path.parent / f"{uid}_metadata.json"
            metadata = Metadata.from_json(metadata_json_path)
            file_list.add_file(metadata)
        return file_list

# JSON helper for serializing Path objects
def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
