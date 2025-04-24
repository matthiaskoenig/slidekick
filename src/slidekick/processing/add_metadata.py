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
        stain_name = console.input("Stain name: ")
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
    from slidekick import DATA_PATH, OUTPUT_PATH
    from slidekick.io import import_wsi
    from slidekick.metadata import FileList
    from slidekick.console import console
    from dataclasses import asdict

    """Full pipeline: create metadata for each file, save all, reload and verify."""

    file_list = FileList()

    supported_suffixes = {".ndpi", ".qptiff", ".tiff", ".czi"}
    image_paths = [p for p in DATA_PATH.glob("*") if p.suffix in supported_suffixes]

    for image_path in image_paths:
        try:
            console.print(f"[bold green]Processing image:[/bold green] {image_path.name}")

            # Define the path where the converted/stored file would go (same name for simplicity)
            stored_path = OUTPUT_PATH / image_path.name

            image_data, metadata = import_wsi(image_path)

            metadata.set_image_type("fluorescence")
            metadata.set_stains({0: "DAPI", 1: "FITC"})
            metadata.set_annotations({"note": f"Test annotation for {image_path.name}"})

            # Add to file list
            file_list.add_file(metadata)

        except Exception as e:
            console.print(f"[error]Failed to process {image_path.name}: {e}")
            continue

    # Save all metadata and file list
    file_list.save_all(OUTPUT_PATH)

    # Load file list back
    file_list_json = OUTPUT_PATH / "file_list.json"
    reloaded_file_list = FileList.from_json(file_list_json)

    # Check integrity of the saved and reloaded metadata
    for uid in file_list.files:
        original = asdict(file_list.get_file(uid))
        reloaded = asdict(reloaded_file_list.get_file(uid))

        # Convert paths to strings for comparison
        original["path_original"] = str(original["path_original"])
        original["path_storage"] = str(original["path_storage"])
        reloaded["path_original"] = str(reloaded["path_original"])
        reloaded["path_storage"] = str(reloaded["path_storage"])

        # Compare dictionaries and print the difference if they don't match
        if original != reloaded:
            print(f"[error] Metadata mismatch for UID {uid}")
            for key in original.keys():
                if original[key] != reloaded[key]:
                    print(f"Difference in '{key}':")
                    print(f"  Original: {original[key]}")
                    print(f"  Reloaded: {reloaded[key]}")
            # Optionally raise the error after printing the differences
            raise AssertionError(f"Metadata mismatch for UID {uid}")

    print(f"\n[✓] Metadata pipeline test passed for {len(file_list.files)} files.")
