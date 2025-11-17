from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from slidekick import OUTPUT_PATH
from slidekick.processing.baseoperator import BaseOperator
from slidekick.io.metadata import Metadata


# Convert input to 2D float32 grayscale
def _as_gray_f32(a, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    elif a.ndim == 3 and a.shape[-1] > 1:
        a = a.mean(axis=-1)
    if a.ndim != 2:
        raise ValueError(f"{name}: expected 2D array, got {a.shape}")
    if a.dtype == object or np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.integer):
        return a.astype(np.float32)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32, copy=False)
    return a.astype(np.float32)


# Grouped stats per stain and bin
def _agg(s: pd.Series) -> pd.Series:
    a = s.to_numpy(dtype=np.float32, copy=False)
    n = np.sum(~np.isnan(a))
    if n == 0:
        return pd.Series(
            {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "p01": np.nan,
                "p10": np.nan,
                "p50": np.nan,
                "p90": np.nan,
                "p99": np.nan,
            },
            dtype="float32",
        )
    return pd.Series(
        {
            "count": int(n),
            "mean": float(np.nanmean(a)),
            "std": float(np.nanstd(a)),
            "min": float(np.nanmin(a)),
            "max": float(np.nanmax(a)),
            "p01": float(np.nanpercentile(a, 1)),
            "p10": float(np.nanpercentile(a, 10)),
            "p50": float(np.nanpercentile(a, 50)),
            "p90": float(np.nanpercentile(a, 90)),
            "p99": float(np.nanpercentile(a, 99)),
        },
        dtype="float32",
    )


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in s)


class LobuleStatisticsOperator(BaseOperator):
    """
    Compute stain intensity statistics as a function of portality.

    Parameters
    ----------
    portality_meta : Metadata
        Float image in [0, 1] encoding portality per pixel.
    stain_metas : Sequence[Metadata]
        One or more co-registered stain images. Any numeric dtype.
    num_bins : int, optional
        Number of uniform bins on [0, 1].
    base_level : int, optional
        Pyramid level index to use when the reader returns a multi-level image.
    out_subdir : str, optional
        Subdirectory under OUTPUT_PATH where results and plots are written.
    """

    def __init__(
        self,
        portality_meta: Metadata,
        stain_metas: Sequence[Metadata],
        *,
        num_bins: int = 20,
        base_level: int = 0,
        out_subdir: str = "lobule_statistics",
    ) -> None:
        # Normalize storage to files like other operators (Path-backed)
        self.portality_meta = portality_meta
        self.stain_metas = stain_metas

        self.num_bins = int(num_bins)
        self.base_level = int(base_level)
        self.out_subdir = str(out_subdir)

        metas = [self.portality_meta] + list(self.stain_metas)
        super().__init__(metas, channel_selection=None)

    def apply(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the operator.

        Returns
        -------
        raw_df : pandas.DataFrame
            Per-pixel table with portality, bin index and stain intensities.
        summary_df : pandas.DataFrame
            Summary statistics per stain and bin with portality bin metadata.
        """
        # Load portality at requested level
        portality = self.load_image(0)[self.base_level]

        # Load stains at requested level
        stain_list = [self.load_image(i + 1)[self.base_level] for i in range(len(self.stain_metas))]

        # Output directory
        outdir = Path(OUTPUT_PATH) / self.out_subdir
        outdir.mkdir(parents=True, exist_ok=True)

        # Valid mask
        p = np.asarray(portality, dtype=np.float32)
        valid = np.isfinite(p)

        # Portality bins
        edges = np.linspace(0.0, 1.0, self.num_bins + 1, dtype=np.float32)
        centers = (edges[:-1] + edges[1:]) * 0.5
        # Clip full image, then take the valid part
        p_clip = np.clip(p, 0.0, 1.0, out=np.empty_like(p, dtype=np.float32))
        xc = p_clip[valid]  # 1D array of clipped portality at valid pixels
        bin_idx = np.searchsorted(edges, xc, side="right") - 1
        np.minimum(bin_idx, len(edges) - 2, out=bin_idx)  # ensure 1.0 -> last bin
        np.maximum(bin_idx, 0, out=bin_idx)

        # Stain column names derived from metadata where possible
        stain_names: List[str] = []
        for i, m in enumerate(self.stain_metas):
            for key in ("label", "name", "stain", "channel_name"):
                if hasattr(m, key):
                    v = getattr(m, key)
                    if isinstance(v, str) and v:
                        stain_names.append(v)
                        break
            else:
                stain_names.append(f"stain_{i}")

        # Collect per-stain vectors aligned to the valid mask and build raw_df
        # 1) normalize shapes and dtypes
        portality = _as_gray_f32(portality, "portality")
        H, W = portality.shape
        stain_list = [_as_gray_f32(a, f"stain[{i}]") for i, a in enumerate(stain_list)]
        for j, a in enumerate(stain_list):
            if a.shape != (H, W):
                raise ValueError(f"Shape mismatch for stain[{j}]: {a.shape} vs portality {(H, W)}")

        # 2) build stain cube (S, H, W) with NaNs where non-finite
        if len(stain_list):
            stains_clean = [np.where(np.isfinite(a), a, np.float32(np.nan)) for a in stain_list]
            stain_cube = np.stack(stains_clean, axis=0)  # (S, H, W)
            stain_matrix = stain_cube.reshape(len(stain_list), -1)  # (S, H*W)
        else:
            stain_matrix = np.empty((0, H * W), dtype=np.float32)

        # 3) flatten once using the existing valid mask and vectors
        mask_flat = np.asarray(valid, dtype=bool).ravel()
        if stain_matrix.size:
            stain_flat = stain_matrix[:, mask_flat].T  # (N_valid, S)
        else:
            stain_flat = np.empty((mask_flat.sum(), 0), dtype=np.float32)

        # 4) assemble DataFrame from precomputed 'xc' and 'bin_idx' plus stains
        raw_df = pd.DataFrame(
            {
                "portality": xc.astype(np.float32, copy=False),  # already clipped and valid-only
                "bin": bin_idx.astype(np.int16, copy=False),  # valid-only
            }
        )
        for k, name in enumerate(stain_names):
            raw_df[name] = stain_flat[:, k] if stain_flat.size else np.empty((0,), dtype=np.float32)

        # Long format: one row per pixel per stain
        long_df = raw_df.melt(
            id_vars=["portality", "bin"],
            value_vars=stain_names,
            var_name="stain",
            value_name="intensity",
        )

        stats = (
            long_df.groupby(["stain", "bin"], sort=True)["intensity"]
            .apply(_agg)
            .reset_index()
        )

        # Bin edge metadata
        bin_left = edges[:-1]
        bin_right = edges[1:]
        bins_df = pd.DataFrame(
            {
                "bin": np.arange(self.num_bins, dtype=np.int16),
                "portality_left": bin_left.astype(np.float32, copy=False),
                "portality_right": bin_right.astype(np.float32, copy=False),
                "portality_center": centers.astype(np.float32, copy=False),
            }
        )

        summary_df = (
            stats.merge(bins_df, on="bin", how="left")
            .sort_values(["stain", "bin"], kind="mergesort")
            .reset_index(drop=True)
        )

        # Plots

        # Plot 1: Mean + std per stain as continuous lines
        stats_ms = summary_df[summary_df["level_2"].isin(["mean", "std"])]
        stats_ms_wide = (
            stats_ms
            .pivot_table(
                index=["stain", "bin", "portality_center"],
                columns="level_2",
                values="intensity",
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for stain in stain_names:
            df_stain = stats_ms_wide[stats_ms_wide["stain"] == stain]
            if df_stain.empty:
                continue
            x = df_stain["portality_center"].to_numpy(dtype=np.float32, copy=False)
            y = df_stain["mean"].to_numpy(dtype=np.float32, copy=False)
            y_std = df_stain["std"].to_numpy(dtype=np.float32, copy=False)
            ax.plot(x, y, label=stain)
            if np.any(np.isfinite(y_std)):
                ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Portality")
        ax.set_ylabel("Intensity")
        ax.set_title("Mean and standard deviation by portality")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "lobule_stats_mean_std.png", dpi=150)
        plt.close(fig)

        # Plot 2: Box plots per bin and stain, one plot per stain
        for stain in stain_names:
            df_raw = raw_df[["bin", stain]].copy()
            df_raw = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
            if df_raw.empty:
                continue

            grouped = df_raw.groupby("bin")[stain]
            bin_indices = np.array(sorted(grouped.groups.keys()), dtype=int)
            if bin_indices.size == 0:
                continue

            data_per_bin = [
                grouped.get_group(b).to_numpy(dtype=np.float32, copy=False)
                for b in bin_indices
            ]

            centers_for_bins = centers[bin_indices]
            if centers_for_bins.ndim == 0:
                centers_for_bins = centers_for_bins[None]

            if edges.size > 1:
                bin_width = float(edges[1] - edges[0])
            else:
                bin_width = 1.0
            width = bin_width * 0.8

            fig_box, ax_box = plt.subplots(figsize=(6.0, 4.0))
            ax_box.boxplot(
                data_per_bin,
                positions=centers_for_bins,
                widths=width,
                manage_ticks=False,
            )
            ax_box.set_xlim(0.0, 1.0)
            ax_box.set_xlabel("Portality")
            ax_box.set_ylabel("Intensity")
            ax_box.set_title(f"Intensity by portality bin (boxplot) – {stain}")
            fig_box.tight_layout()
            fig_box.savefig(
                outdir / (f"lobule_stats_boxplot_{_slug(stain)}.png"),
                dpi=150,
            )
            plt.close(fig_box)

        # Plot 3: Violin plot per bin and stain, one plot per stain
        for stain in stain_names:
            df_raw = raw_df[["bin", stain]].copy()
            df_raw = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
            if df_raw.empty:
                continue

            grouped = df_raw.groupby("bin")[stain]
            bin_indices = np.array(sorted(grouped.groups.keys()), dtype=int)
            if bin_indices.size == 0:
                continue

            data_per_bin = [
                grouped.get_group(b).to_numpy(dtype=np.float32, copy=False)
                for b in bin_indices
            ]

            centers_for_bins = centers[bin_indices]
            if centers_for_bins.ndim == 0:
                centers_for_bins = centers_for_bins[None]

            if edges.size > 1:
                bin_width = float(edges[1] - edges[0])
            else:
                bin_width = 1.0
            width = bin_width * 0.8

            fig_violin, ax_violin = plt.subplots(figsize=(6.0, 4.0))
            ax_violin.violinplot(
                data_per_bin,
                positions=centers_for_bins,
                widths=width,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            ax_violin.set_xlim(0.0, 1.0)
            ax_violin.set_xlabel("Portality")
            ax_violin.set_ylabel("Intensity")
            ax_violin.set_title(f"Intensity by portality bin (violin) – {stain}")
            fig_violin.tight_layout()
            fig_violin.savefig(
                outdir / (f"lobule_stats_violin_{_slug(stain)}.png"),
                dpi=150,
            )
            plt.close(fig_violin)

        # Write outputs
        raw_df.to_csv(outdir / "lobule_stats_raw.csv", index=False)
        summary_df.to_csv(outdir / "summarized_stats.csv", index=False)

        return raw_df, summary_df


if __name__ == "__main__":
    # Example usage for manual testing
    from slidekick import DATA_PATH
    from slidekick.processing.lobule_segmentation.lobule_segmentation import LobuleSegmentor

    image_paths = [
        DATA_PATH / "reg_n_sep" / "noise.tiff",
        DATA_PATH / "reg_n_sep" / "periportal.tiff",
        DATA_PATH / "reg_n_sep" / "perivenous.tiff",
    ]

    metadata_for_segmentation = [
        Metadata(path_original=Path(image_path), path_storage=Path(image_path)) for image_path in image_paths
    ]

    segmentor = LobuleSegmentor(
        metadata_for_segmentation,
        channels_pp=1,
        channels_pv=2,
        base_level=0,
        region_size=25,
        adaptive_histonorm=True,
    )

    metadata_segmentation, metadata_portality = segmentor.apply()

    # Run the LobuleStatisticsOperator (pooled across lobules)
    operator = LobuleStatisticsOperator(metadata_portality, metadata_for_segmentation, num_bins=10)
    operator.apply()

    print(f"Lobule statistics saved in: {OUTPUT_PATH / 'lobule_statistics'}")
