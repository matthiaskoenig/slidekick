import os
from pathlib import Path
from typing import Tuple, Annotated

import numpy as np
import tifffile
import zarr

from slidekick.console import console
from slidekick.metadata import Metadata

def add_metadata(image_data, metadata: Metadata) -> None:
    """
    Interactively annotate and add metadata to a Metadata object using the console.
    Prompts for image type and stain information.
    """
    # Display image for annotation on visual basis
    # TODO: Add visualization

    # Use console to get image type information
    # pre-defined options: fluorescence, immunohistochemistry, brightfield
    # last option other with free text input


    console.print("\n[info]Annotating image metadata...[/]")

    # Select image type
    console.print("Select image type:", style="info")
    console.print("1. Fluorescence\n2. Immunohistochemistry\n3. Brightfield\n4. Other", style="info")
    choice = console.input("Enter choice [1-4]: ")
    if choice == "1":
        image_type = "fluorescence"
    elif choice == "2":
        image_type = "immunohistochemistry"
    elif choice == "3":
        image_type = "brightfield"
    elif choice == "4":
        image_type = console.input("Enter custom image type: ")
    else:
        console.print("[warning]Invalid choice, defaulting to 'other'[/]")
        image_type = console.input("Enter image type: ")
    metadata.set_image_type(image_type)
    console.print(f"[success]Image type set to '{image_type}'[/]")

    # Use console to get stain information in image

    # Gather stain information
    console.print("\n[info]Stain information[/]")
    console.print("Select stain mode:", style="info")
    console.print("1. Single stain as color (RGB or similar)\n2. Different stains per channel (multiplex)\n3. Other",
                  style="info")
    mode = console.input("Enter choice [1-3]: ")
    stains = {}
    if mode == "1":
        stain_name = console.input(f"Stain name: ")
        stains[0] = stain_name
    elif mode == "2":
        count = console.input("Number of channels: ")
        try:
            num = int(count)
        except ValueError:
            console.print("[error]Invalid number, defaulting to 1 channel[/]")
            num = 1
        for idx in range(num):
            stain_name = console.input(f"Stain name for channel {idx}: ")
            stains[idx] = stain_name
    elif mode == "3":
        raw = console.input("Enter stain information: ")
        stains[0] = raw
    else:
        console.print("[warning]Invalid choice, using 'Other' raw input[/]")
        raw = console.input("Enter stain information: ")
        stains[0] = raw

    metadata.set_stains(stains)
    console.print(f"[success]Stains set: {stains}[/]")


if __name__ == "__main__":
    from slidekick import DATA_PATH
    from slidekick.io import wsi

    # Read whole-slide image
    # image_path = DATA_PATH / "SIM-22-034_4plex.qptiff"
    image_path = DATA_PATH / "NOR-021_CYP1A2.ndpi"
    # image_path = DATA_PATH / "APAP_HE.czi"

    # Load image and auto-generated metadata
    image_data, meta_data = wsi.read_wsi(image_path)
    console.print("Loaded image data:", image_data)
    console.print("Initial metadata:", meta_data)

    # Convert to numpy array if needed
    image_array: np.ndarray = np.array(image_data)

    # Run interactive metadata annotation
    add_metadata(image_data, meta_data)

    # Display updated metadata
    console.print("[success]Final metadata:[/]")
    console.print(meta_data)




