import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import uuid
from datetime import datetime
from pathlib import Path
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

        # Convert string keys back to integers in the 'stains' dictionary
        if "stains" in data:
            data["stains"] = {int(k): v for k, v in data["stains"].items()}

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
