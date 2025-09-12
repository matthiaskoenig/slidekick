import numpy as np
from pathlib import Path
from rich.prompt import Confirm
from typing import List, Union, Optional, Tuple, Dict, Any
import cv2
import datetime
import uuid
from sklearn.cluster import KMeans
import napari

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from slidekick.processing.baseoperator import BaseOperator
from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH
from slidekick.io import save_tif

from slidekick.processing.roi.roi_utils import largest_bbox, ensure_grayscale_uint8, crop_image, detect_tissue_mask
from slidekick.processing.lobule_segmentation.get_segments import segment_thinned_image
from slidekick.processing.lobule_segmentation.process_segments import process_segments_to_mask
from slidekick.processing.lobule_segmentation.lob_utils import (
    detect_tissue_mask_multiotsu, overlay_mask, pad_image, build_mask_pyramid_from_processed,
    downsample_to_max_side, percentile, gray_for_cluster, render_cluster_gray
)


class LobuleSegmentor(BaseOperator):
    """
    Based on https://github.com/matthiaskoenig/zonation-image-analysis/blob/develop/src/zia/pipeline/pipeline_components/segementation_component.py
    """

    def __init__(self,
                 metadata: Union[Metadata, List[Metadata]],
                 channel_selection: Union[int, List[int]] = None,
                 channels_pp: Union[int, List[int]] = None,
                 channels_pv: Union[int, List[int]] = None,
                 throw_out_ratio: float = None,
                 preview: bool = True,
                 confirm: bool = True,
                 base_level: int = 0,
                 region_size: int = 20,
                 multi_otsu: bool = True,
                 ksize: int = 7,
                 n_clusters: int = 3,
                 alpha_pv: float = 2.0,
                 alpha_pp: float = 2.0,
                 target_superpixels: int = None,
                 # vessel gating (always on)
                 min_vessel_area_pp: int = 1500,
                 min_vessel_area_pv: int = 1000,
                 vessel_annulus_px: int = 3,
                 vessel_zone_ratio_thr_pp: float = 0.75,
                 vessel_zone_ratio_thr_pv: float = 0.60,
                 vessel_circularity_min: float = 0.15,
                 vessel_contrast_min_pp: float = 10.0,
                 vessel_contrast_min_pv: float = 6.0,
                 vessel_bg_fraction_min: float = 0.20,
                 # refinement knobs (always on)
                 dist_midband_rel: float = 0.15,
                 dist_margin_px: int = 2,
                 refine_pp_percentile: float = 0.30,
                 refine_pv_percentile: float = 0.30):
        """
        @param metadata: List or single metadata object to load and use for lobule segmentation. All objects used should be single channel and either periportal or pericentrally expressed.
        @param channel_selection: Number of Metadata objects that should be inverted. Channels with stronger, i.e., brighter expression / absorpotion perincentrally should be inverted. If None, invert none.
        @param channels_pp: Channel[s] that are brighter in periportal regions (potentially after inversion). At least one channel has to be set for channel_pp or channel_pv. If only one is provided, the other will be calculated as the furthest distance on the same channel(s).
        @param channels_pv: Channel[s] that are brighter in periveneous regions (potentially after inversion). At least one channel has to be set for channel_pp or channel_pv. If only one is provided, the other will be calculated as the furthest distance on the same channel(s).
        @param throw_out_ratio: the min percentage of non_background pixel for the slide to have to not be discarded
        @param preview: whether to preview the segmentation
        @param confirm: whether to confirm the segmentation
        @param base_level: Pyramid level to load
        @param multi_otsu: whether to use multi-otsu to filter out microscopy background, otherwise classic otsu
        @param region_size: average size of superpixels for SLIC in pixels
        @param ksize: kernel size for convolution in filtering (cv2.median blur), must be odd and greater than 1, e.g., 3, 5, 7, ...
        @param n_clusters: number of clusters in (weighted) K-Means to use for superpixel clustering
        @param alpha_pv: weighting factors for superpixel clustering for pv channels
        @param alpha_pp: weighting factors for superpixel clustering for pp channels
        @param target_superpixels: number of superpixels to use for segmentation, overrides region_size
        @param min_vessel_area_pp: min area (px) for PP-enclosed vessel candidates
        @param min_vessel_area_pv: min area (px) for PV-enclosed vessel candidates
        @param vessel_annulus_px: thickness (px) of ring-consistency check
        @param vessel_zone_ratio_thr_pp: fraction of ring that must be PP
        @param vessel_zone_ratio_thr_pv: fraction of ring that must be PV
        @param vessel_circularity_min: 4πA/P^2 gate; low keeps elongated vessels
        @param vessel_contrast_min_pp: PP ring − lumen contrast (uint8) in PP channels
        @param vessel_contrast_min_pv: PV ring − lumen contrast (uint8) in PV channels
        @param vessel_bg_fraction_min: share of inside SPs that were background before reassignment
        @param dist_midband_rel: relative band where |d_cv - d_pf| small → MID assignment
        @param dist_margin_px: absolute pixel margin for PP/PV distance decisions
        @param refine_pp_percentile: demote weakest PP to MID below this PP-channel percentile
        @param refine_pv_percentile: demote weakest PV to MID below this PV-channel percentile
        """
        self.throw_out_ratio = throw_out_ratio
        self.preview = preview
        self.confirm = confirm
        self.base_level = base_level
        self.multi_otsu = multi_otsu
        self.region_size = region_size
        self.ksize = ksize
        self.n_clusters = n_clusters
        self.alpha_pv = alpha_pv
        self.alpha_pp = alpha_pp
        self.target_superpixels = target_superpixels

        # vessel gating
        self.min_vessel_area_pp = min_vessel_area_pp
        self.min_vessel_area_pv = min_vessel_area_pv
        self.vessel_annulus_px = vessel_annulus_px
        self.vessel_zone_ratio_thr_pp = vessel_zone_ratio_thr_pp
        self.vessel_zone_ratio_thr_pv = vessel_zone_ratio_thr_pv
        self.vessel_circularity_min = vessel_circularity_min
        self.vessel_contrast_min_pp = vessel_contrast_min_pp
        self.vessel_contrast_min_pv = vessel_contrast_min_pv
        self.vessel_bg_fraction_min = vessel_bg_fraction_min

        # refinement knobs
        self.dist_midband_rel = dist_midband_rel
        self.dist_margin_px = dist_margin_px
        self.refine_pp_percentile = refine_pp_percentile
        self.refine_pv_percentile = refine_pv_percentile

        # make sure channel is list for later iteration
        if isinstance(channel_selection, int):
            channel_selection = [channel_selection]
        elif channel_selection is None:
            channel_selection = []
        super().__init__(metadata, channel_selection=None)
        # Override init back to have direct info on channels to invert
        self.channels = channel_selection
        # Make lists
        if isinstance(channels_pp, int):
            channels_pp = [channels_pp]
        if isinstance(channels_pv, int):
            channels_pv = [channels_pv]
        # check that the defined channels exist in the metadata objects
        if (
                (channels_pp is not None and any(i > len(metadata) - 1 for i in channels_pp)) or
                (channels_pv is not None and any(i > len(metadata) - 1 for i in channels_pv))
        ):
            console.print(
                "At least one channel for pp or pv is greater than number of metadata objects.",
                style="error"
            )
            raise ValueError(
                "At least one channel for pp or pv is greater than number of metadata objects."
            )
        # Check that at least one of the lists is not None:
        if channels_pp is None and channels_pv is None:
            console.print(f"No values for channel_pp or channel_pv provided.", style="error")
            raise ValueError(f"At least one channel hast to be set in either channel_pp or channel_pv.")
        elif channels_pp is not None and channels_pv is not None:
            # Check that channels_pp and channels_pv are different
            overlap = set(channels_pp) & set(channels_pv)
            if overlap:
                console.print(
                    f"The following IDs are used both for periportal and perivenous detection: {overlap}. This is not possible.",
                    style="error")
                raise ValueError(f"Identical elements for channels_pp and channels_pv: {overlap}")
        self.channels_pp = channels_pp
        self.channels_pv = channels_pv

    def _filter(self, image_stack: np.ndarray) -> np.ndarray:
        """
        Channel-wise version of zia's filtering. Works for N >= 1 channels.
        Keeps dtype uint8, returns (H, W, N).
        """

        if image_stack.ndim != 3:
            raise ValueError(f"Expected (H, W, N) stack, got {image_stack.shape}")

        H, W, N = image_stack.shape

        # set missing to 0 (any zero across channels -> zero all channels)
        missing_mask = np.any(image_stack == 0, axis=-1)

        out = np.empty((H, W, N), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

        for c in range(N):
            ch = image_stack[..., c]

            # ensure uint8 for OpenCV ops
            if ch.dtype != np.uint8:
                ch = np.clip(ch, 0, 255).astype(np.uint8)

            # apply missing mask on this channel
            ch[missing_mask] = 0

            # blur, ksize from zia=7
            ch = cv2.medianBlur(ch, ksize=self.ksize)

            # adaptive histogram norm (2D only)
            ch = clahe.apply(ch)
            ch[ch < 10] = 0  # suppress background altered by CLAHE

            # blur
            ch = cv2.medianBlur(ch, ksize=3)

            # channel normalization to 99th percentile
            # use >0 to avoid zero-inflation from background
            nz = ch[ch > 0]
            p99 = float(np.percentile(nz, 99)) if nz.size else 255.0
            if p99 <= 0:
                p99 = 1.0
            ch = (ch.astype(np.float32) / p99 * 255.0).clip(0, 255).astype(np.uint8)

            out[..., c] = ch

        return out

    def _preview_images(self, images: List[np.ndarray], titles: Optional[List[str]] = None) -> None:
        """
        Display the provided images in a Napari viewer as separate layers.
        Each image will be added as a grayscale or RGBA layer (with appropriate naming).
        Napari is launched only if preview is True.
        """
        if not images:
            console.print("No images to preview.", style="warning")
            return

        # Downsample each image for preview to a reasonable size (max 2048 px on longest side)
        thumbs = [downsample_to_max_side(im, 2048) for im in images]

        # Import Napari and create a viewer for layered display
        # TODO: Use viewer from slidekick
        viewer = napari.Viewer()

        # Add each image as a separate layer in the viewer
        for idx, im in enumerate(thumbs):
            # Determine a layer name using provided titles or default naming
            layer_name = titles[idx] if titles and idx < len(titles) else f"Layer_{idx}"
            # Add as grayscale or color layer depending on image shape
            if im.ndim == 2:
                viewer.add_image(im, name=str(layer_name), colormap='gray')
            elif im.ndim == 3 and im.shape[-1] in (3, 4):
                # If image has 3 or 4 channels, treat it as an RGB(A) image
                viewer.add_image(im, name=str(layer_name), rgb=True)
            else:
                # Fallback: add image with default settings
                viewer.add_image(im, name=str(layer_name))

        # Start the Napari event loop (blocks until all viewer windows are closed)
        napari.run()

    def _load_and_invert_images_from_metadatas(self, report_path: Path = None) -> [np.ndarray, Tuple[int, int, int, int], Dict[int, Any]]:
        """
        Load each image at the pyramid level defined, invert selected channels,
        harmonize shapes, and stack along the last axis -> (H, W, N).
        """

        arrays = []

        for i, md in enumerate(self.metadata):
            try:
                img = self.load_image(i)[self.base_level]  # Load resolution
            except:
                raise Exception(f"Level {self.base_level} failed for image {i} in stack")
            # ensure 2D
            img = np.asarray(img)
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                img = np.mean(img[..., :3], axis=-1).astype(img.dtype)

            arrays.append(img)

        if not arrays:
            raise RuntimeError("No images loaded after filtering.")

        # stack along the third dimension
        # harmonize shapes before stacking
        h_min = min(a.shape[0] for a in arrays)
        w_min = min(a.shape[1] for a in arrays)
        arrays = [a[:h_min, :w_min] for a in arrays]

        stack = np.dstack(arrays).astype(np.uint8)

        # Get resolutions / image sizes as dict for later back mapping based on metadata object 0
        orig_shapes = {k: v.shape[:2] for k, v in self.load_image(0).items()}

        # ROI detection block
        console.print("Complete. Detecting ROI...", style="info")
        stack_grayscale = ensure_grayscale_uint8(stack)
        if self.multi_otsu:
            # We detect three levels in each wsi: actual tissue, black background and the background in microscopy.
            # returns tissue_mask
            tissue_mask = detect_tissue_mask_multiotsu(stack_grayscale, morphological_radius=6, report_path=report_path)
        else:
            # just tissue detection
            tissue_mask = detect_tissue_mask(stack_grayscale, morphological_radius=6)

        # stack: (H,W,C); tissue_mask: (H,W) True=tissue
        stack[~tissue_mask] = 0

        # crop to tissue bbox
        bbox = largest_bbox(tissue_mask.astype(np.uint8))
        stack = crop_image(stack, bbox)

        # Inversion and discard only AFTER cropping:
        for i in range(stack.shape[-1]):

            # optional inversion
            if self.channels is not None and i in self.channels:
                if np.issubdtype(stack[:, :, i].dtype, np.integer):
                    info = np.iinfo(stack[:, :, i].dtype)
                    stack[:, :, i] = info.max - stack[:, :, i]
                else:  # float images assumed in [0,1]
                    stack[:, :, i] = 1.0 - stack[:, :, i]

            # optional discard by foreground coverage
            if self.throw_out_ratio is not None:
                non_bg = np.count_nonzero(img != 255)
                ratio = non_bg / stack[:, :, i].size
                if ratio < self.throw_out_ratio:
                    console.print(
                        f"Discarded {i} with non-background pixel ratio: {ratio:.3f}",
                        style="warning",
                    )
                    stack = np.delete(stack, i, axis=-1)
                    if i in self.channels_pp or i in self.channels_pv:
                        console.print(
                            f"Channel used for periportal or perivenous detection removed.",
                            style="warning",
                        )
                        if i in self.channels_pp:
                            self.channels_pp.remove(i)
                        else:
                            self.channels_pv.remove(i)

                        # Check if channels_pp and channels_pv are not empty lists now
                        if not self.channels_pp and not self.channels_pv:
                            console.print("No remaining channels for perivenous or periportal detection available.", style="error")
                            raise Exception(f"No remaining channels for segmentation available.")

                    # IDs in channels_pp and channels_pv have to be adapted if channel is removed
                    self.channels_pp = [j-1 for j in self.channels_pp if j >= i]
                    self.channels_pv = [j-1 for j in self.channels_pv if j >= i]

        return stack, bbox, orig_shapes


    def skeletize_kmeans(self, image_stack: np.ndarray,
                         pad=10,
                         region_size=6,
                         report_path: Path = None) -> Tuple[np.ndarray, Tuple[List[int], list]]:
        """
        Adopts most code from zia.../clustering.py. Uses intensities of pre-defined stain-channels.
        We cluster superpixels, label centers (PP/MID/PV), then search for ring-like PP/PV regions
        and reassign enclosed superpixels as vessels to the surrounding class. Saves zonation overlay
        with vessel outlines (central=cyan, portal=magenta), then thinning. Adds refinement:
        distance-based filling and histogram cutoffs guided by vessels.
        """
        # 1) Superpixel generation
        image_stack = pad_image(image_stack, pad)

        console.print("Generating superpixels...", style="info")
        superpixelslic = cv2.ximgproc.createSuperpixelSLIC(
            image_stack, algorithm=cv2.ximgproc.MSLIC, region_size=region_size
        )
        superpixelslic.iterate(num_iterations=20)

        superpixel_mask = superpixelslic.getLabelContourMask(thick_line=False)
        labels = superpixelslic.getLabels()
        num_labels = superpixelslic.getNumberOfSuperpixels()

        if report_path is not None:
            cv2.imwrite(str(report_path / "superpixels.png"), superpixel_mask)

        merged = image_stack.astype(float)
        super_pixels = {label: merged[labels == label] for label in range(num_labels)}

        # 2) Foreground/background split based on dark fraction across channels
        background_pixels: Dict[int, np.ndarray] = {}
        foreground_pixels: Dict[int, np.ndarray] = {}

        console.print("Complete. Clustering superpixels into foreground and background pixels...", style="info")
        for label, pixels in super_pixels.items():
            channel_dark_fraction = (pixels <= 0).sum(axis=0) / pixels.shape[0]
            if (channel_dark_fraction > 0.5).any():
                background_pixels[label] = pixels
            else:
                foreground_pixels[label] = pixels

        fg_keys = list(foreground_pixels.keys())
        fg_pix_mask = np.isin(labels, np.asarray(fg_keys, dtype=np.int32))
        foreground_pixels_means = {
            lab: np.mean(arr, axis=0).astype(np.float32) for lab, arr in foreground_pixels.items()
        }

        if report_path is not None:
            out_template = np.zeros(shape=(image_stack.shape[0], image_stack.shape[1], 3), dtype=np.uint8)
            for i in range(num_labels):
                if i in background_pixels:
                    out_template[labels == i] = np.array([0, 0, 0], dtype=np.uint8)
                else:
                    out_template[labels == i] = np.array([255, 255, 255], dtype=np.uint8)
            cv2.imwrite(str(report_path / "superpixels_bg_fg.png"), out_template)

            for k in range(merged.shape[2]):
                out_gray = np.zeros(shape=(image_stack.shape[0], image_stack.shape[1]), dtype=np.float32)
                for i, mean_vec in foreground_pixels_means.items():
                    out_gray[labels == i] = mean_vec[k]
                m = float(out_gray.max())
                if m > 0:
                    out_gray = (out_gray / m) * 255.0
                out_u8 = out_gray.astype(np.uint8)
                out_rgb = cv2.cvtColor(out_u8, cv2.COLOR_GRAY2RGB)
                mask_on = (np.all(out_rgb != [0, 0, 0], axis=2)) & (superpixel_mask == 255)
                out_rgb[mask_on] = [255, 255, 0]
                cv2.imwrite(str(report_path / f"superpixel_mean_{k}.png"), out_rgb)

            X_means = np.stack([foreground_pixels_means[k] for k in fg_keys], axis=0)
            C = X_means.shape[1]
            fig, axes = plt.subplots(1, C, figsize=(2.8 * C, 3.0), squeeze=False, constrained_layout=True)
            for c in range(C):
                axes[0, c].hist(X_means[:, c], bins=50)
                axes[0, c].set_title(f"Ch {c} · mean")
                axes[0, c].set_xlabel("value")
                axes[0, c].set_ylabel("count")
            fig.suptitle("Superpixel mean intensity histograms", fontsize=11)
            fig.savefig(str(report_path / "hist_superpixel_means.png"), dpi=150)
            plt.close(fig)

        # 3) Weighted KMeans over superpixel mean features
        console.print(f"Complete. Cluster (n={self.n_clusters}) the foreground superpixels based on superpixel mean values...",
                      style="info")

        X = np.stack([foreground_pixels_means[k] for k in fg_keys], axis=0)
        C = X.shape[1]

        # TODO: Evaluate non-linear weighting

        w_feat = np.ones(C, dtype=np.float32)
        if self.channels_pp is not None:
            w_feat[self.channels_pp] *= self.alpha_pp
        if self.channels_pv is not None:
            w_feat[self.channels_pv] *= self.alpha_pv
        Xw = X * w_feat

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        kmeans.fit(Xw)

        centers = kmeans.cluster_centers_ / w_feat
        labels_k = kmeans.labels_

        # Assign semantic roles PP/MID/PV
        channels_pp = self.channels_pp
        channels_pv = self.channels_pv
        if (channels_pp is not None) and (channels_pv is not None):
            pp_scores = centers[:, channels_pp].mean(axis=1)
            pv_scores = centers[:, channels_pv].mean(axis=1)
            idx_pp = int(np.argmax(pp_scores))
            pv_order = np.argsort(-pv_scores)
            idx_pv = int(pv_order[0] if pv_order[0] != idx_pp else pv_order[1])
        elif channels_pp is not None:
            s = centers[:, channels_pp].mean(axis=1)
            idx_pp = int(np.argmax(s))
            idx_pv = int(np.argmin(s))
        else:
            s = centers[:, channels_pv].mean(axis=1)
            idx_pv = int(np.argmax(s))
            idx_pp = int(np.argmin(s))

        idx_mid = [i for i in range(self.n_clusters) if i not in [idx_pp, idx_pv]]
        sorted_label_idx = np.array([idx_pp, *idx_mid, idx_pv], dtype=int)

        # Map each foreground superpixel id -> cluster id
        superpixel_kmeans_map = {sp: km for sp, km in zip(fg_keys, labels_k)}

        # Build initial assignment dict for all superpixels (background: -1)
        assigned_by_sp: Dict[int, int] = {sp: -1 for sp in range(num_labels)}
        for sp, km in superpixel_kmeans_map.items():
            assigned_by_sp[sp] = km

        # Vectorized lookup of assigned_by_sp over the SLIC label image
        get_lab = np.vectorize(lambda sp: assigned_by_sp.get(int(sp), -1))
        cluster_map = get_lab(labels).astype(np.int32)  # shape (H, W), values in {-1, 0..n_clusters-1}

        if report_path is not None:
            template_init = render_cluster_gray(cluster_map, sorted_label_idx, self.n_clusters)
            template_init[~fg_pix_mask] = 0
            cv2.imwrite(str(report_path / "foreground_clustered_initial.png"), template_init)

        console.print("Complete. Now detecting vessels in detected zonation...", style="info")

        # Vessel detection and superpixel reassignment
        vessel_classes: List[int] = []
        vessel_contours: List[np.ndarray] = []

        def _reassign_hole_contours(zone_cluster_idx: int,
                                    mask_zone: np.ndarray,
                                    vessel_cls_value: int):
            """
            zone_cluster_idx: idx_pp or idx_pv (surrounding zone)
            mask_zone: binary mask (255=zone pixels)
            vessel_cls_value: 1 for portal field (PP), 0 for central vein (PV)
            """
            contours, hierarchy = cv2.findContours(mask_zone, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if contours is None or hierarchy is None:
                return

            Hh = hierarchy[0]
            H, W = mask_zone.shape

            # thresholds per zone (always on; from __init__)
            if zone_cluster_idx == idx_pp:
                min_area = int(self.min_vessel_area_pp)
                zone_ratio_thr = float(self.vessel_zone_ratio_thr_pp)
                min_contrast = float(self.vessel_contrast_min_pp)
                ring_channels = (self.channels_pp or [])
            else:
                min_area = int(self.min_vessel_area_pv)
                zone_ratio_thr = float(self.vessel_zone_ratio_thr_pv)
                min_contrast = float(self.vessel_contrast_min_pv)
                ring_channels = (self.channels_pv or [])

            k = max(1, int(self.vessel_annulus_px))
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))

            def _mean_over_channels(mask_bool: np.ndarray, channels: List[int]) -> float:
                if not channels or not np.any(mask_bool):
                    return 0.0
                vals = [image_stack[..., c][mask_bool].mean() for c in channels]
                return float(np.mean(vals)) if vals else 0.0

            for i, cnt in enumerate(contours):
                # only child contours (holes)
                if Hh[i][3] == -1:
                    continue

                area = cv2.contourArea(cnt)
                if area < max(1, min_area):
                    continue

                # shape sanity (permissive to allow elongated vessels)
                peri = max(cv2.arcLength(cnt, True), 1e-6)
                circularity = float(4.0 * np.pi * area / (peri * peri))
                if circularity < float(self.vessel_circularity_min):
                    continue

                # hole mask
                hole_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(hole_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                hole_bool = hole_mask > 0

                # ring (annulus) around the hole
                dil = cv2.dilate(hole_mask, se)
                ero = cv2.erode(hole_mask, se)
                ring = cv2.subtract(dil, ero)
                ring_bool = ring > 0
                if not np.any(ring_bool):
                    continue

                # ring must be mostly the same zone
                zone_ratio = float(np.mean((cluster_map == zone_cluster_idx)[ring_bool]))
                if zone_ratio < zone_ratio_thr:
                    continue

                # intensity contrast in defining channels: ring brighter than hole
                if ring_channels:
                    ring_mean = _mean_over_channels(ring_bool, ring_channels)
                    hole_mean = _mean_over_channels(hole_bool, ring_channels)
                    if (ring_mean - hole_mean) < min_contrast:
                        continue

                # accept: reassign all superpixels whose pixels fall inside the hole
                sp_inside = np.unique(labels[hole_bool])
                for sp in sp_inside:
                    assigned_by_sp[int(sp)] = zone_cluster_idx

                vessel_classes.append(vessel_cls_value)
                vessel_contours.append(cnt)

        # PP holes -> portal fields; PV holes -> central veins
        mask_pp = (cluster_map == idx_pp).astype(np.uint8) * 255
        _reassign_hole_contours(idx_pp, mask_pp, vessel_cls_value=1)

        mask_pv = (cluster_map == idx_pv).astype(np.uint8) * 255
        _reassign_hole_contours(idx_pv, mask_pv, vessel_cls_value=0)

        # Rebuild final per-pixel cluster map after reassignment
        cluster_map_final = get_lab(labels).astype(np.int32)

        console.print("Complete. Now refining zonation...", style="info")

        # Refinement stage: distance-based filling + histogram cutoffs
        Hh, Wh = cluster_map_final.shape
        portal_mask = np.zeros((Hh, Wh), dtype=np.uint8)
        central_mask = np.zeros((Hh, Wh), dtype=np.uint8)
        for cnt, cls in zip(vessel_contours, vessel_classes):
            if cls == 1:   # portal field
                cv2.drawContours(portal_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            else:          # central vein
                cv2.drawContours(central_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # distance to vessels (pixels = float32 distance in px)
        if np.any(portal_mask):  # cv2.distanceTransform expects 0 as target -> invert
            dist_pf = cv2.distanceTransform(255 - portal_mask, cv2.DIST_L2, 5)
        else:
            dist_pf = np.full((Hh, Wh), 1e6, dtype=np.float32)
        if np.any(central_mask):
            dist_cv = cv2.distanceTransform(255 - central_mask, cv2.DIST_L2, 5)
        else:
            dist_cv = np.full((Hh, Wh), 1e6, dtype=np.float32)

        # fill background (-1) based on nearest vessel with a margin; mid when close
        bg_mask = (cluster_map_final < 0) & fg_pix_mask
        if np.any(bg_mask):
            delta = dist_cv - dist_pf  # negative -> closer to central
            mid_band = np.abs(delta) <= (self.dist_midband_rel * np.minimum(dist_cv, dist_pf) + self.dist_margin_px)
            assign_pv = (delta < -self.dist_margin_px) & (~mid_band)
            assign_pp = (delta > self.dist_margin_px) & (~mid_band)

            mid_choice = idx_mid[0] if len(idx_mid) else idx_pp
            cluster_map_final[bg_mask & mid_band] = mid_choice
            cluster_map_final[bg_mask & assign_pp] = idx_pp
            cluster_map_final[bg_mask & assign_pv] = idx_pv

        # histogram cutoffs to trim weak PP/PV at borders (demote to MID)

        # PP demotion
        if self.channels_pp is not None and len(idx_mid):
            pp_zone = (cluster_map_final == idx_pp)
            if np.any(pp_zone):
                pp_vals = np.mean(np.stack([image_stack[..., c] for c in self.channels_pp], axis=2), axis=2)
                thr_pp = percentile(pp_vals[pp_zone], self.refine_pp_percentile)
                demote_pp = pp_zone & (pp_vals < thr_pp)
                cluster_map_final[demote_pp] = idx_mid[0]

        # PV demotion
        if self.channels_pv is not None and len(idx_mid):
            pv_zone = (cluster_map_final == idx_pv)
            if np.any(pv_zone):
                pv_vals = np.mean(np.stack([image_stack[..., c] for c in self.channels_pv], axis=2), axis=2)
                thr_pv = percentile(pv_vals[pv_zone], self.refine_pv_percentile)
                demote_pv = pv_zone & (pv_vals < thr_pv)
                cluster_map_final[demote_pv] = idx_mid[0]

        template = render_cluster_gray(cluster_map_final, sorted_label_idx, self.n_clusters)
        template[~fg_pix_mask] = 0
        if report_path is not None:
            cv2.imwrite(str(report_path / "foreground_clustered_final.png"), template)

        # Zonation + vessel overlay
        if report_path is not None:
            base_vis = image_stack.mean(axis=2).astype(np.uint8)
            base_rgb = cv2.cvtColor(base_vis, cv2.COLOR_GRAY2BGR)

            COLOR_PP = (255, 0, 255)
            COLOR_MID = (60, 160, 60)
            COLOR_PV = (0, 165, 255)

            zonation_rgb = np.zeros_like(base_rgb, dtype=np.uint8)
            mask_pp = (cluster_map_final == idx_pp) & fg_pix_mask
            mask_pv = (cluster_map_final == idx_pv) & fg_pix_mask
            mask_mid = np.zeros_like(mask_pp, dtype=bool)
            for mid_idx in idx_mid:
                mask_mid |= (cluster_map_final == mid_idx)
            mask_mid &= fg_pix_mask

            zonation_rgb[mask_pp] = COLOR_PP
            zonation_rgb[mask_mid] = COLOR_MID
            zonation_rgb[mask_pv] = COLOR_PV

            overlay = cv2.addWeighted(base_rgb, 0.6, zonation_rgb, 0.4, 0.0)

            # outlines always on: central=cyan, portal=magenta
            for cnt, cls in zip(vessel_contours, vessel_classes):
                color = (255, 255, 0) if cls == 0 else (255, 0, 255)
                cv2.drawContours(overlay, [cnt], -1, color, thickness=2)

            cv2.imwrite(str(report_path / "zonation_overlay.png"), overlay)
            cv2.imwrite(str(report_path / "zonation_rgb.png"), zonation_rgb)

        # TODO: Thinning

        console.print("Complete. Run thinning algorithm...", style="info")

        # TODO: Draw final map + vessels

        template = cv2.medianBlur(255 - template, 5)

        console.print("Complete. Run thinning algorithm...", style="info")

        thinned = cv2.ximgproc.thinning(template.reshape(template.shape[0], template.shape[1], 1).astype(np.uint8))

        # TODO: Double thinning + vessels

        if report_path is not None:
            cv2.imwrite(str(report_path / "thinned.png"), thinned)

        return thinned, (vessel_classes, vessel_contours)


    def apply(self) -> Metadata:

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        new_uid = f"{timestamp}-{short_id}"

        report_path = OUTPUT_PATH / f"segmentation-{new_uid}"
        report_path.mkdir(parents=True, exist_ok=True)

        console.print("Loading images...", style="info")
        # Load images and invert based on metadata
        img_stack, bbox, orig_shapes = self._load_and_invert_images_from_metadatas(report_path)

        img_size_base = img_stack.shape[:2]  # base size for back-mapping of mask

        console.print("Complete. Now applying filters...", style="info")

        # Apply filters
        img_stack = self._filter(img_stack)

        # Show Preview after loading and filtering
        if self.preview:
            # Ensure we have a list of arrays to preview
            # numpy array -> split into list
            if isinstance(img_stack, list):
                imgs = img_stack
            elif hasattr(img_stack, "shape"):
                if img_stack.ndim == 2:
                    imgs = [np.asarray(img_stack)]
                elif img_stack.ndim == 3:
                    # assume H, W, N and split channels
                    imgs = [np.asarray(img_stack[..., i]) for i in range(img_stack.shape[2])]
                else:
                    raise ValueError("Unsupported img_stack shape for preview.")

            # Optional titles from metadata if available
            titles = []
            for i, md in enumerate(self.metadata):
                name = getattr(md, "name", None) or getattr(md, "sample_id", None) \
                       or getattr(md, "path", None)
                try:
                    name = Path(name).name  # shorten if it's a path
                except Exception:
                    pass
                titles.append(str(name) if name is not None else f"image_{i}")

            self._preview_images(imgs, titles)

        if self.confirm:
            apply = Confirm.ask("Continue with lobule segmentation?", default=True, console=console)
            if not apply:
                console.print("Aborted by user. No segmentation performed.", style="error")
                return self.metadata

        # Superpixel algorithm (steps 3–7)
        pad = 10  # adjust if you want a different safety border

        # Adjust region size from number of superpixels if specified
        if self.target_superpixels is not None:
            H, W, _ = img_stack.shape
            region_size = int(np.sqrt((H * W) / self.target_superpixels))
            console.print(f"Region size for SLIC was computed as {region_size}", style="info")
            if region_size < self.region_size:
                console.print(f"Computed region size {region_size} is smaller than specified region size {self.region_size}", style="warning")
            self.region_size = region_size

        thinned, (vessel_classes, vessel_contours) = self.skeletize_kmeans(
            img_stack,
            pad=pad,
            region_size=self.region_size,
            report_path=report_path,
        )

        console.print("Complete. Creating lines segments from skeleton...", style="info")

        # Steps 8: segments
        line_segments = segment_thinned_image(thinned)

        console.print("Complete. Creating segmentation mask...", style="info")

        # Step 9: Creating lobule and vessel polygons from line segments and vessel contours
        mask = process_segments_to_mask(line_segments, thinned.shape, report_path=report_path)

        # Crop mask by padding
        mask_cropped = mask[pad:-pad, pad:-pad]

        # Preview
        # Left: Orig, Middle: Overlay, Right: Mask only
        if self.preview:
            # Original as mean projection
            orig_vis = img_stack.mean(axis=2).astype(np.uint8)
            # Skeleton cropped to match mask/img_stack
            overlay_vis = overlay_mask(img_stack, mask_cropped, alpha=0.5)
            # Mask visualization as binary 0/255 for clarity
            mask_vis = (mask_cropped > 0).astype(np.uint8) * 255
            self._preview_images([orig_vis, overlay_vis, mask_vis], titles=["Original", "Segmentation Overlay", "Mask"])

        console.print("Complete. Back-mapping mask to all pyramid levels...", style="info")

        # Build masks for every level (steps 1–3)
        mask_pyramid = build_mask_pyramid_from_processed(
            mask_cropped=mask_cropped,
            img_size_base=img_size_base,  # ROI size at base_level (after bbox crop)
            bbox_base=bbox,  # bbox in base_level coords used for the crop
            orig_shapes=orig_shapes,  # {level: (H,W)} from self.load_image(0)
            base_level=self.base_level,
        )

        console.print("Complete.", style="info")

        # TODO: package `full_mask` as a new Metadata and save

        new_meta = Metadata(
            path_original=report_path,
            path_storage=report_path,
            image_type="mask",
            uid=new_uid,
        )

        new_meta.save(report_path)
        # Pass the full level -> array dict so save_tif writes a tiled, pyramidal TIFF
        save_tif(mask_pyramid, report_path / f"{new_uid}_seg.tiff", metadata=new_meta)

        return new_meta



if __name__ == "__main__":
    # Example usage
    from slidekick import DATA_PATH

    image_paths = [DATA_PATH / "reg_n_sep" / "noise.tiff",
                   DATA_PATH / "reg_n_sep" / "periportal.tiff",
                   DATA_PATH / "reg_n_sep" / "perivenous.tiff",
                   ]

    metadata_for_segmentation = [Metadata(path_original=image_path, path_storage=image_path) for image_path in image_paths]

    Segmentor = LobuleSegmentor(metadata_for_segmentation,
                                channels_pp=1,
                                channels_pv=2,
                                base_level=1,
                                region_size=25)

    metadata_segmentation = Segmentor.apply()
