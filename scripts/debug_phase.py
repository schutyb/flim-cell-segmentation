#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import tifffile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from skimage import morphology, segmentation, measure

from phasorpy.phasor import phasor_from_signal
from phasorpy.lifetime import phasor_calibrate
from phasorpy.filter import phasor_filter_median


# -----------------------
# IO helpers
# -----------------------
def load_tiff_hyx(path: Path) -> np.ndarray:
    """Return (H,Y,X). Robust for most FLIM TIFF stacks."""
    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        return arr[0]
    raise ValueError(f"Unexpected TIFF shape {arr.shape} for {path}")


def bin_xy_sum(signal: np.ndarray, bin: int) -> np.ndarray:
    """Spatial binning by summing photons in (Y,X). Expects (H, Y, X)."""
    H, Y, X = signal.shape
    Y2 = (Y // bin) * bin
    X2 = (X // bin) * bin
    sig = signal[:, :Y2, :X2]
    return sig.reshape(H, Y2 // bin, bin, X2 // bin, bin).sum(axis=(2, 4))


def parse_tile_index(p: Path) -> int:
    m = re.search(r"Im_(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def assemble_snake(tiles, nrows, ncols, snake_order=True):
    h0, w0 = tiles[0].shape[:2]
    tiles = [t[:h0, :w0, ...] for t in tiles]
    rows = []
    idx = 0
    for rr in range(nrows):
        row_tiles = tiles[idx: idx + ncols]
        if snake_order and (rr % 2 == 1):
            row_tiles = row_tiles[::-1]
        rows.append(np.concatenate(row_tiles, axis=1))
        idx += ncols
    return np.concatenate(rows, axis=0)


# -----------------------
# Visualization
# -----------------------
def make_phase_pseudocolor_abs(
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    mean_min: float,
    phase_cmap: mcolors.Colormap,
    v_percentile: float = 99.5,
    gamma: float = 0.9,
) -> np.ndarray:
    """
    Pseudocolor with ABSOLUTE phase mapping:
      - color = atan2(S,G) mapped from [-pi, pi] -> [0,1]
      - brightness = mean normalized by percentile
    """
    mask = np.isfinite(real) & np.isfinite(imag) & np.isfinite(mean) & (mean > mean_min)
    phase = np.arctan2(imag, real)

    phase_norm = (phase + np.pi) / (2 * np.pi)  # [-pi,pi] -> [0,1]
    phase_norm = np.clip(phase_norm, 0, 1)
    phase_norm[~mask] = 0.0

    rgb = phase_cmap(phase_norm)[..., :3].astype(np.float32)

    mean_f = mean.astype(np.float32)
    if np.any(mask):
        vmax = np.nanpercentile(mean_f[mask], v_percentile)
        vmax = max(vmax, 1e-12)
    else:
        vmax = 1.0

    val = np.clip(mean_f / vmax, 0, 1)
    if gamma and gamma > 0:
        val = val ** gamma
    val[~mask] = 0.0

    rgb *= val[..., None]
    return np.clip(rgb, 0, 1)


def overlay_boundaries(base_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    b = segmentation.find_boundaries(labels > 0, mode="outer")
    out = base_rgb.copy()
    out[b] = 1.0
    return out


def save_phase_debug(out_png: Path, mean, real, imag, labels, mean_min=3.0):
    phase = np.arctan2(imag, real)
    mask = np.isfinite(phase) & np.isfinite(mean) & (mean > mean_min)
    b = segmentation.find_boundaries(labels > 0, mode="outer")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(mean, cmap="gray")
    ax[0].set_title("mean (binned)")
    ax[0].axis("off")

    im = ax[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax[1].set_title("phase = atan2(S,G) [rad]")
    ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    phase_rgb = plt.cm.twilight((phase + np.pi) / (2 * np.pi))[..., :3]
    phase_rgb[~mask] = 0
    phase_rgb[b] = 1.0
    ax[2].imshow(phase_rgb)
    ax[2].set_title("phase + segmentation boundaries")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_phasor_scatter(out_png: Path, mean, real, imag, labels, mean_min=3.0):
    g = real.ravel()
    s = imag.ravel()
    m = mean.ravel()
    lab = labels.ravel()

    valid = np.isfinite(g) & np.isfinite(s) & np.isfinite(m) & (m > mean_min)

    plt.figure(figsize=(6, 6))
    plt.scatter(g[valid], s[valid], s=1, c="lightgray", alpha=0.25)

    sel = valid & (lab > 0)
    if np.any(sel):
        plt.scatter(g[sel], s[sel], s=3, c="lime", alpha=0.85)

    t = np.linspace(0, 1, 400)
    plt.plot(t, np.sqrt(np.clip(t - t**2, 0, None)), color="black", lw=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal")
    plt.xlabel("G")
    plt.ylabel("S")
    plt.title("Phasor scatter (selected pixels in green)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------
# Baseline segmentation (for debugging only)
# -----------------------
def segment_baseline(mean, real, imag,
                     mean_min=3.0,
                     bright_q=90.0,
                     phase_abs_min=-np.pi,
                     phase_abs_max=np.pi,
                     min_area=4,
                     max_area=60):
    """
    Simple baseline:
      - pick bright pixels (percentile)
      - (optionally) restrict ABSOLUTE phase range
      - keep small blobs
    This is NOT final segmentation, just to overlay and debug.
    """
    valid = np.isfinite(mean) & np.isfinite(real) & np.isfinite(imag) & (mean > mean_min)
    if not np.any(valid):
        return np.zeros_like(mean, dtype=np.int32)

    phase = np.arctan2(imag, real)

    T = np.nanpercentile(mean[valid], bright_q)
    cand = valid & (mean >= T) & (phase >= phase_abs_min) & (phase <= phase_abs_max)

    # minimal cleaning
    cand = morphology.remove_small_objects(cand, min_size=min_area)
    labels = measure.label(cand)

    # remove too large
    if labels.max() > 0 and max_area is not None:
        props = measure.regionprops(labels)
        rm = [p.label for p in props if p.area > max_area]
        if rm:
            labels[np.isin(labels, rm)] = 0
            labels = measure.label(labels > 0)

    return labels.astype(np.int32)


def main():
    # -----------------------
    # HARD-CODED paths (your setup)
    # -----------------------
    folder = Path("/Users/schutyb/Documents/cell_segmentation/dod/data_raw/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp")
    ref_path = Path(
        "/Users/schutyb/Documents/cell_segmentation/dod/data_raw/p449/visit01/coumarin6_ExVivo_slide"
        "/Image03_FOV256_z152_32Sp/Im_00001.tif"
    )

    outdir = Path("/Users/schutyb/Documents/cell_segmentation/dod/data_process/p449/visit01/debug_phase")
    (outdir / "tiles").mkdir(parents=True, exist_ok=True)
    (outdir / "mosaic").mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Params
    # -----------------------
    bin = 4
    nfirst = 16
    frequency_mhz = 80.0
    ref_lifetime_ns = 2.5

    do_filter = True
    filter_size = 3
    filter_repeat = 2

    # pseudocolor
    mean_min = 3.0
    v_percentile = 99.5
    gamma = 0.9
    spectral_colors = ["#6A0DAD", "#0000FF", "#00FF00", "#FFFF00", "#FFA500", "#FF0000"]
    phase_cmap = mcolors.LinearSegmentedColormap.from_list("custom_spectral", spectral_colors, N=256)

    # baseline seg (debug knobs)
    bright_q = 90.0
    # start with no phase restriction; after you inspect debug_phase.png,
    # set a narrower absolute range, e.g. phase_abs_min=0.45, phase_abs_max=0.70
    phase_abs_min = -np.pi
    phase_abs_max = np.pi
    min_area = 4
    max_area = 60

    # mosaic layout
    nrows, ncols = 4, 4
    snake_order = True

    # -----------------------
    # tiles
    # -----------------------
    tile_paths = sorted(folder.glob("Im_*.tif"), key=parse_tile_index)
    tile_paths = tile_paths[: nrows * ncols]
    if not tile_paths:
        raise SystemExit(f"No Im_*.tif found in {folder}")

    # -----------------------
    # reference phasor
    # -----------------------
    ref_signal = load_tiff_hyx(ref_path)[:nfirst]
    ref_binned = bin_xy_sum(ref_signal, bin)
    ref_mean, ref_real, ref_imag = phasor_from_signal(ref_binned, axis=0)

    pseudo_tiles = []
    overlay_tiles = []

    for i, tp in enumerate(tile_paths, start=1):
        tile_name = tp.stem
        print(f"[{i}/{len(tile_paths)}] {tile_name}")

        try:
            signal = load_tiff_hyx(tp)[:nfirst]
        except Exception as e:
            print(f"  !! skip: {e}")
            continue

        sig_binned = bin_xy_sum(signal, bin)
        mean, real, imag = phasor_from_signal(sig_binned, axis=0)

        # calibration
        real, imag = phasor_calibrate(
            real, imag,
            ref_mean, ref_real, ref_imag,
            frequency=frequency_mhz,
            lifetime=ref_lifetime_ns,
        )

        # filtering
        if do_filter:
            mean, real, imag = phasor_filter_median(mean, real, imag, size=filter_size, repeat=filter_repeat)

        # pseudocolor (ABS phase mapping)
        rgb = make_phase_pseudocolor_abs(
            mean, real, imag,
            mean_min=mean_min,
            phase_cmap=phase_cmap,
            v_percentile=v_percentile,
            gamma=gamma,
        )
        pseudo_tiles.append(rgb)
        plt.imsave(outdir / "tiles" / f"{tile_name}_pseudocolor_abs.png", rgb)

        # baseline segmentation (debug)
        labels = segment_baseline(
            mean, real, imag,
            mean_min=mean_min,
            bright_q=bright_q,
            phase_abs_min=phase_abs_min,
            phase_abs_max=phase_abs_max,
            min_area=min_area,
            max_area=max_area,
        )

        ov = overlay_boundaries(rgb, labels)
        overlay_tiles.append(ov)
        plt.imsave(outdir / "tiles" / f"{tile_name}_overlay.png", ov)

        # debug plots
        save_phase_debug(outdir / "tiles" / f"{tile_name}_debug_phase.png",
                         mean, real, imag, labels, mean_min=mean_min)

        save_phasor_scatter(outdir / "tiles" / f"{tile_name}_phasor_debug.png",
                            mean, real, imag, labels, mean_min=mean_min)

    # mosaics
    if len(pseudo_tiles) >= nrows * ncols:
        mosaic_rgb = assemble_snake(pseudo_tiles[: nrows * ncols], nrows, ncols, snake_order=snake_order)
        plt.imsave(outdir / "mosaic" / "mosaic_pseudocolor_abs.png", mosaic_rgb)

    if len(overlay_tiles) >= nrows * ncols:
        mosaic_ov = assemble_snake(overlay_tiles[: nrows * ncols], nrows, ncols, snake_order=snake_order)
        plt.imsave(outdir / "mosaic" / "mosaic_overlay.png", mosaic_ov)

    print(f"Done. Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()