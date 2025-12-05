import slidekick
from slidekick import DATA_PATH
from slidekick.io.metadata import Metadata
from slidekick.processing import ValisRegistrator, StainNormalizer, StainSeparator, LobuleSegmentor, LobuleStatistics

if __name__ == '__main__':

    image_paths = [DATA_PATH / "reg" / "HE1.ome.tif",
                   DATA_PATH / "reg" / "HE2.ome.tif",
                   DATA_PATH / "reg" / "Arginase1.ome.tif",
                   DATA_PATH / "reg" / "KI67.ome.tif",
                   DATA_PATH / "reg" / "GS_CYP1A2.czi",
                   DATA_PATH / "reg" / "Ecad_CYP2E1.czi",
                   ]

    slidekick.OUTPUT_PATH = DATA_PATH / "slidekick"

    # Create Metadata
    # TODO: Populate metadata
    metadatas = [Metadata(path_original=image_path, path_storage=image_path) for image_path in image_paths]

    # Normalize HE Stains
    metadatas_normalized = [StainNormalizer(metadata, channel_selection=[0, 1, 2]).apply() for metadata in metadatas[:2]]

    metadatas_normalized += metadatas[2:]

    # Align MetaDatas with VALIS
    Registrator = ValisRegistrator(metadatas_normalized, max_processed_image_dim_px=600, max_non_rigid_registration_dim_px=600,
                                   confirm=False)

    metadatas_registered = Registrator.apply()

    # Separate Stains
    metadatas_separated = [StainSeparator(metadata, mode="fluorescence", confirm=False, preview=False).apply()
                           for metadata in metadatas_registered if "CYP" in metadata.filename_original]

    # Choose metadata for segmentation
    # metadatas_registered: [HE1, HE2, Arginase1, KI67, GS_CYP1A2, ECAD_CYP1A2]
    # metadatas_separated: [[GS, CYP1A2, DAPI], [Ecad, CYP2E1, DAPI]]
    metadatas_for_segmentation = [metadatas_registered[2],  # Arginase1 -> PP
                                  metadatas_separated[0][0],  # GS -> PV Marker
                                  metadatas_separated[1][0],  # Ecad -> PP
                                  ]

    # Segment Lobules
    segmentor = LobuleSegmentor(metadatas_for_segmentation, channels_pp=[0,2], channels_pv=1, base_level=2,
                                region_size=200, adaptive_histonorm=True)

    metadata_segmentation, metadata_portality = segmentor.apply()

    # Create Lobule Statistics
    metadata_for_stats = [metadatas_registered[2:4],  # Arginase1, KI67
                          metadatas_separated[0][:2],  # GS, CYP1A2,
                          metadatas_separated[1][:2],  # ECad, CYP2E1
                          ]
    operator = LobuleStatistics(metadata_portality, metadata_for_stats, num_bins=10)
    operator.apply()