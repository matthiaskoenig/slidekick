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


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in s)


def load_distributions(npz_path) -> dict:
    """Load a ``distributions.npz`` saved by :class:`LobuleStatistics`.

    Returns a plain dict with keys:

    ``hist_counts``       (S, num_bins, n_hist_bins) int64 — raw pixel counts
    ``hist_edges``        (S, n_hist_bins+1) float32 — intensity bin edges per stain
    ``stain_names``       list[str]
    ``portality_centers`` (num_bins,) float32 — bin centre portality values
    ``portality_edges``   (num_bins+1,) float32

    Example — reconstruct percentiles for stain 0, portality bin 5::

        d = load_distributions("lobule_statistics/distributions.npz")
        counts = d["hist_counts"][0, 5]           # histogram for (stain0, bin5)
        edges  = d["hist_edges"][0]
        cdf    = np.cumsum(counts) / counts.sum()
        p50    = edges[np.searchsorted(cdf, 0.50)]
    """
    data = np.load(npz_path, allow_pickle=True)
    return {
        "hist_counts":       data["hist_counts"],
        "hist_edges":        data["hist_edges"],
        "stain_names":       list(data["stain_names"]),
        "portality_centers": data["portality_centers"],
        "portality_edges":   data["portality_edges"],
    }


class LobuleStatistics(BaseOperator):
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
        **Key speedup for large WSI:** level 1 = 2× downsample (4× fewer pixels),
        level 2 = 4× downsample (16× fewer pixels).  Statistics are nearly identical
        at lower resolution.
    out_subdir : str, optional
        Subdirectory under OUTPUT_PATH where results and plots are written.
    n_hist_bins : int, optional
        Number of intensity bins for the per-portality-bin histograms saved in
        ``distributions.npz`` (default 200).  These histograms replace the old
        per-pixel raw CSV: they are compact (≈ 50 KB for 3 stains × 20 bins),
        allow exact violin/boxplot reproduction, and support arbitrary percentile
        queries without reprocessing the image.
    write_raw_csv : bool, optional
        Write the per-pixel ``raw_df`` to CSV (default ``False``).  Only useful
        for small images; at WSI scale the file is several GB.
    """

    def __init__(
        self,
        portality_meta: Metadata,
        stain_metas: Sequence[Metadata],
        *,
        num_bins: int = 20,
        base_level: int = 0,
        out_subdir: str = "lobule_statistics",
        n_hist_bins: int = 200,
        write_raw_csv: bool = False,
    ) -> None:
        self.portality_meta = portality_meta
        self.stain_metas = stain_metas
        self.num_bins = int(num_bins)
        self.base_level = int(base_level)
        self.out_subdir = str(out_subdir)
        self.n_hist_bins = int(n_hist_bins)
        self.write_raw_csv = bool(write_raw_csv)

        metas = [self.portality_meta] + list(self.stain_metas)
        super().__init__(metas, channel_selection=None)

    def apply(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the operator.

        Returns
        -------
        raw_df : pandas.DataFrame
            Per-pixel table with portality, bin index and stain intensities
            (one row per valid pixel, one column per stain).
        summary_df : pandas.DataFrame
            Wide-form summary statistics per (stain, bin):
            count, mean, std, min, max, p01, p10, p50, p90, p99,
            portality_left, portality_right, portality_center.
        """
        # ── Load images ──────────────────────────────────────────────────────
        portality = self.load_image(0)[self.base_level]
        stain_list = [self.load_image(i + 1)[self.base_level] for i in range(len(self.stain_metas))]

        outdir = Path(OUTPUT_PATH) / self.out_subdir
        outdir.mkdir(parents=True, exist_ok=True)

        # ── Valid mask & portality binning ────────────────────────────────────
        p = np.asarray(portality, dtype=np.float32)
        valid = np.isfinite(p)

        edges = np.linspace(0.0, 1.0, self.num_bins + 1, dtype=np.float32)
        centers = (edges[:-1] + edges[1:]) * 0.5

        # Clip into a temporary buffer (out= avoids one allocation); then work
        # only on the valid subset from this point forward.
        p_clip = np.clip(p, 0.0, 1.0, out=np.empty_like(p))
        xc = p_clip[valid]          # 1-D, valid pixels only
        del p, p_clip               # free full-res portality

        bin_idx = np.searchsorted(edges, xc, side="right") - 1
        np.minimum(bin_idx, len(edges) - 2, out=bin_idx)
        np.maximum(bin_idx, 0, out=bin_idx)

        # ── Stain column names ────────────────────────────────────────────────
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

        # ── Normalise and validate stain shapes ───────────────────────────────
        portality_arr = _as_gray_f32(portality, "portality")
        H, W = portality_arr.shape
        del portality_arr           # no longer needed

        stain_list = [_as_gray_f32(a, f"stain[{i}]") for i, a in enumerate(stain_list)]
        for j, a in enumerate(stain_list):
            if a.shape != (H, W):
                raise ValueError(f"Shape mismatch for stain[{j}]: {a.shape} vs portality {(H, W)}")

        # ── Build stain_flat (N_valid, S) — skip the (S, H, W) cube ──────────
        # Writing each stain directly into a pre-allocated slice avoids the
        # intermediate (S, H, W) → reshape → transpose chain (≈ 1 extra full
        # copy of all stain data eliminated).
        mask_flat = valid.ravel()
        N_valid = int(mask_flat.sum())
        S = len(stain_list)

        if S > 0:
            stain_flat = np.empty((N_valid, S), dtype=np.float32)
            for s_idx, a in enumerate(stain_list):
                arr = np.where(np.isfinite(a), a, np.float32(np.nan))
                stain_flat[:, s_idx] = arr.ravel()[mask_flat]
        else:
            stain_flat = np.empty((N_valid, 0), dtype=np.float32)
        del stain_list              # free per-stain full-res arrays

        # ── raw_df ────────────────────────────────────────────────────────────
        raw_df = pd.DataFrame({
            "portality": xc.astype(np.float32, copy=False),
            "bin": bin_idx.astype(np.int16, copy=False),
        })
        for k, name in enumerate(stain_names):
            raw_df[name] = stain_flat[:, k] if S else np.empty((0,), dtype=np.float32)

        # ── Statistics + plot data in a single pass ───────────────────────────
        # Replaces: melt → S × N_valid long DataFrame
        #           groupby.apply(_agg) → 5 separate nanpercentile sorts per group
        # New:      one loop over (stain, bin), one np.percentile call per group,
        #           plot data collected in the same pass (no re-groupby for plots).
        stats_rows = []
        plot_data = {}   # stain -> (bin_indices_arr, [vals_array_per_bin])

        for s_idx, stain_name in enumerate(stain_names):
            stain_col = stain_flat[:, s_idx]    # view, no copy
            bins_present: List[int] = []
            data_bins: List[np.ndarray] = []

            for b in range(self.num_bins):
                vals = stain_col[bin_idx == b]
                vals = vals[np.isfinite(vals)]
                n = int(vals.size)
                if n == 0:
                    stats_rows.append({
                        "stain": stain_name, "bin": b, "count": 0,
                        "mean": np.nan, "std": np.nan,
                        "min": np.nan, "max": np.nan,
                        "p01": np.nan, "p10": np.nan, "p50": np.nan,
                        "p90": np.nan, "p99": np.nan,
                    })
                else:
                    # Single sort for all 5 percentiles (was 5 separate sorts).
                    p01, p10, p50, p90, p99 = np.percentile(vals, [1, 10, 50, 90, 99])
                    stats_rows.append({
                        "stain": stain_name, "bin": b, "count": n,
                        "mean": float(vals.mean()),
                        "std":  float(vals.std()),
                        "min":  float(vals.min()),
                        "max":  float(vals.max()),
                        "p01": float(p01), "p10": float(p10), "p50": float(p50),
                        "p90": float(p90), "p99": float(p99),
                    })
                    bins_present.append(b)
                    data_bins.append(vals)

            plot_data[stain_name] = (np.array(bins_present, dtype=int), data_bins)

        # ── summary_df ────────────────────────────────────────────────────────
        stats_wide = pd.DataFrame(stats_rows)

        bins_df = pd.DataFrame({
            "bin":               np.arange(self.num_bins, dtype=np.int16),
            "portality_left":    edges[:-1].astype(np.float32, copy=False),
            "portality_right":   edges[1:].astype(np.float32, copy=False),
            "portality_center":  centers.astype(np.float32, copy=False),
        })

        summary_df = (
            stats_wide.merge(bins_df, on="bin", how="left")
            .sort_values(["stain", "bin"], kind="mergesort")
            .reset_index(drop=True)
        )

        # ── Plot 1: Mean ± std per stain ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for stain in stain_names:
            df_s = stats_wide[stats_wide["stain"] == stain].merge(bins_df, on="bin", how="left")
            if df_s.empty:
                continue
            x     = df_s["portality_center"].to_numpy(dtype=np.float32, copy=False)
            y     = df_s["mean"].to_numpy(dtype=np.float32, copy=False)
            y_std = df_s["std"].to_numpy(dtype=np.float32, copy=False)
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

        bin_width = float(edges[1] - edges[0]) if edges.size > 1 else 1.0
        width = bin_width * 0.8

        # ── Plot 2: Box plots — pre-computed data, no re-groupby ──────────────
        for stain in stain_names:
            bin_indices, data_per_bin = plot_data[stain]
            if bin_indices.size == 0:
                continue
            centers_for_bins = centers[bin_indices]
            if centers_for_bins.ndim == 0:
                centers_for_bins = centers_for_bins[None]

            fig_box, ax_box = plt.subplots(figsize=(6.0, 4.0))
            ax_box.boxplot(data_per_bin, positions=centers_for_bins,
                           widths=width, manage_ticks=False)
            ax_box.set_xlim(0.0, 1.0)
            ax_box.set_xlabel("Portality")
            ax_box.set_ylabel("Intensity")
            ax_box.set_title(f"Intensity by portality bin (boxplot) – {stain}")
            fig_box.tight_layout()
            fig_box.savefig(outdir / f"lobule_stats_boxplot_{_slug(stain)}.png", dpi=150)
            plt.close(fig_box)

        # ── Plot 3: Violin plots — same pre-computed data ─────────────────────
        for stain in stain_names:
            bin_indices, data_per_bin = plot_data[stain]
            if bin_indices.size == 0:
                continue
            centers_for_bins = centers[bin_indices]
            if centers_for_bins.ndim == 0:
                centers_for_bins = centers_for_bins[None]

            fig_violin, ax_violin = plt.subplots(figsize=(6.0, 4.0))
            ax_violin.violinplot(data_per_bin, positions=centers_for_bins,
                                 widths=width, showmeans=False,
                                 showmedians=True, showextrema=False)
            ax_violin.set_xlim(0.0, 1.0)
            ax_violin.set_xlabel("Portality")
            ax_violin.set_ylabel("Intensity")
            ax_violin.set_title(f"Intensity by portality bin (violin) – {stain}")
            fig_violin.tight_layout()
            fig_violin.savefig(outdir / f"lobule_stats_violin_{_slug(stain)}.png", dpi=150)
            plt.close(fig_violin)

        # ── Histogram NPZ — compact substitute for the per-pixel raw CSV ────────
        # Per (stain, portality_bin): a fixed-grid count array over intensity.
        # Shape: (S, num_bins, n_hist_bins).  Shared edges per stain so percentiles
        # and violin/boxplot data can be reconstructed exactly from the file alone.
        #
        # File size: S × num_bins × n_hist_bins × 4 bytes
        #   e.g. 3 stains × 20 bins × 200 hist-bins = 48 KB   (vs. GB for per-pixel CSV)
        if S > 0:
            hist_counts = np.zeros((S, self.num_bins, self.n_hist_bins), dtype=np.int64)
            hist_edges  = np.empty((S, self.n_hist_bins + 1), dtype=np.float32)
            for s_idx, stain_name in enumerate(stain_names):
                _, data_bins = plot_data[stain_name]
                # Global intensity range for this stain (finite values only).
                all_vals = stain_flat[:, s_idx]
                all_vals = all_vals[np.isfinite(all_vals)]
                if all_vals.size == 0:
                    hist_edges[s_idx] = np.linspace(0.0, 1.0, self.n_hist_bins + 1)
                    continue
                lo, hi = float(all_vals.min()), float(all_vals.max())
                if lo == hi:
                    hi = lo + 1.0
                edges_s = np.linspace(lo, hi, self.n_hist_bins + 1, dtype=np.float32)
                hist_edges[s_idx] = edges_s
                for b_idx, vals in zip(plot_data[stain_name][0], data_bins):
                    counts, _ = np.histogram(vals, bins=edges_s)
                    hist_counts[s_idx, b_idx] = counts

            np.savez_compressed(
                outdir / "distributions.npz",
                hist_counts      = hist_counts,          # (S, num_bins, n_hist_bins)
                hist_edges       = hist_edges,           # (S, n_hist_bins+1)
                stain_names      = np.array(stain_names, dtype=object),
                portality_centers= centers,              # (num_bins,)
                portality_edges  = edges,                # (num_bins+1,)
            )

            # Long-format CSV — one row per (stain, portality_bin, intensity_bin).
            # Columns: stain, portality_bin, portality_center,
            #          intensity_left, intensity_right, intensity_center, count.
            # Rows: S × num_bins × n_hist_bins  (e.g. 3×20×200 = 12 000).
            # Built with numpy tiling — no Python-level loop over rows.
            hist_dfs = []
            for s_idx, stain_name in enumerate(stain_names):
                e = hist_edges[s_idx]               # (n_hist_bins+1,)
                int_left   = e[:-1]                 # (n_hist_bins,)
                int_right  = e[1:]
                int_center = (int_left + int_right) * 0.5
                # Tile so every (portality_bin, intensity_bin) pair has a row.
                # hist_counts[s_idx]: (num_bins, n_hist_bins)
                hist_dfs.append(pd.DataFrame({
                    "stain":            stain_name,
                    "portality_bin":    np.repeat(np.arange(self.num_bins),   self.n_hist_bins),
                    "portality_center": np.repeat(centers,                    self.n_hist_bins),
                    "intensity_left":   np.tile(int_left,   self.num_bins),
                    "intensity_right":  np.tile(int_right,  self.num_bins),
                    "intensity_center": np.tile(int_center, self.num_bins),
                    "count":            hist_counts[s_idx].ravel(),
                }))
            pd.concat(hist_dfs, ignore_index=True).to_csv(
                outdir / "distributions.csv", index=False
            )

        # ── Write outputs ─────────────────────────────────────────────────────
        if self.write_raw_csv:
            raw_df.to_csv(outdir / "lobule_stats_raw.csv", index=False)
        summary_df.to_csv(outdir / "summarized_stats.csv", index=False)

        del stain_flat              # Memory management

        return raw_df, summary_df


if __name__ == "__main__":
    # Example usage for manual testing
    from slidekick import DATA_PATH
    from slidekick.processing.lobule_segmentation import LobuleSegmentor

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

    # Run the LobuleStatistics
    operator = LobuleStatistics(metadata_portality, metadata_for_segmentation, num_bins=10)
    operator.apply()

    print(f"Lobule statistics saved in: {OUTPUT_PATH / 'lobule_statistics'}")
