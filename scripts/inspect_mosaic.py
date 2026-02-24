#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

import matplotlib
matplotlib.use("Agg")  # avoid GUI windows / "ghost" figures
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from phasorpy.phasor import phasor_from_signal
from phasorpy.lifetime import phasor_calibrate


# -----------------------
# Helpers
# -----------------------
def bin_xy_sum(signal: np.ndarray, bin: int) -> np.ndarray:
    """Spatial binning by summing photons in (Y,X). Expects (H, Y, X)."""
    H, Y, X = signal.shape
    Y2 = (Y // bin) * bin
    X2 = (X // bin) * bin
    sig = signal[:, :Y2, :X2]
    return sig.reshape(H, Y2 // bin, bin, X2 // bin, bin).sum(axis=(2, 4))


def make_phase_pseudocolor(
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    mean_min: float,
    phase_cmap: mcolors.Colormap,
    v_percentile: float = 99.5,                # brightness normalization (per tile)
    gamma: float = 0.8,                        # >1 darker, <1 brighter
    phase_percentiles: tuple[float, float] = (1.0, 99.0),  # robust phase range (per tile)
) -> np.ndarray:
    """
    RGB where:
      - hue/color = phase = atan2(S, G), robust-normalized with percentiles
      - brightness = mean intensity, robust-normalized with percentile + gamma
    """
    mask = np.isfinite(real) & np.isfinite(imag) & np.isfinite(mean) & (mean > mean_min)

    # ---- Phase
    phase = np.arctan2(imag, real)

    if np.any(mask):
        pmin, pmax = np.nanpercentile(phase[mask], phase_percentiles)
        if not np.isfinite(pmin) or not np.isfinite(pmax) or (pmax - pmin) < 1e-6:
            pmin, pmax = -np.pi, np.pi
    else:
        pmin, pmax = -np.pi, np.pi

    phase_norm = (phase - pmin) / (pmax - pmin + 1e-12)
    phase_norm = np.clip(phase_norm, 0, 1)
    phase_norm[~mask] = 0.0

    rgb = phase_cmap(phase_norm)[..., :3].astype(np.float32)

    # ---- Brightness (robust)
    mean_f = mean.astype(np.float32)
    if np.any(mask):
        vmax = np.nanpercentile(mean_f[mask], v_percentile)
        vmax = max(vmax, 1e-12)
    else:
        vmax = 1.0

    val = np.clip(mean_f / vmax, 0, 1)
    if gamma is not None and gamma > 0:
        val = val ** gamma
    val[~mask] = 0.0

    rgb *= val[..., None]
    return np.clip(rgb, 0, 1)


def save_rgb(path: Path, rgb: np.ndarray, title: str | None = None) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    # -----------------------
    # PARAMS (hardcoded)
    # -----------------------
    folder = Path(
        "/Users/schutyb/Documents/cell_segmentation/dod/data_raw/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp"
    )

    # Calibration reference (spectral/coumarin) - ALSO use only first 16 channels
    ref_path = Path(
        "/Users/schutyb/Documents/cell_segmentation/dod/data_raw/p449/visit01/coumarin6_ExVivo_slide"
        "/Image03_FOV256_z152_32Sp/Im_00001.tif"
    )

    # Output outside repo (data_process)
    outdir = Path(
        "/Users/schutyb/Documents/cell_segmentation/dod/data_process/p449/visit01/inspect_mosaic_pseudocolor"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # Data/phasor params
    bin = 4
    nfirst = 16
    frequency_mhz = 80.0
    ref_lifetime_ns = 2.5

    # Visualization params
    mean_min = 3.0
    v_percentile = 99.0
    gamma = 0.8
    phase_percentiles = (1.0, 99.0)

    # Mosaic assembly
    grid = "4x4"
    snake_order = True

    # Filtering (to match your earlier “good-looking” output)
    do_filter = True
    filter_size = 3
    filter_repeat = 2

    # -----------------------
    # Colormap: violet -> blue -> green -> yellow -> orange -> red
    # -----------------------
    spectral_colors = ["#6A0DAD", "#0000FF", "#00FF00", "#FFFF00", "#FFA500", "#FF0000"]
    phase_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_spectral", spectral_colors, N=256
    )

    # -----------------------
    # Tiles
    # -----------------------
    tile_paths = sorted(folder.glob("Im_*.tif"))
    if not tile_paths:
        raise SystemExit(f"No Im_*.tif found in: {folder}")

    r, c = grid.lower().split("x")
    nrows, ncols = int(r), int(c)
    expected = nrows * ncols
    if len(tile_paths) < expected:
        print(f"Warning: found {len(tile_paths)} tiles, expected {expected}. Will use {len(tile_paths)}.")
        expected = len(tile_paths)

    # -----------------------
    # Load + prepare reference once (FIRST 16 CHANNELS + binning)
    # -----------------------
    ref_signal = np.asarray(tifffile.imread(str(ref_path)))[:nfirst]
    ref_binned = bin_xy_sum(ref_signal, bin)
    ref_mean, ref_real, ref_imag = phasor_from_signal(ref_binned, axis=0)

    # -----------------------
    # Process tiles -> pseudocolor
    # -----------------------
    pseudo_tiles: list[np.ndarray] = []

    if do_filter:
        from phasorpy.filter import phasor_filter_median

    for i, tp in enumerate(tile_paths[:expected], start=1):
        tile_name = tp.stem
        print(f"[{i}/{expected}] {tile_name}")

        signal = np.asarray(tifffile.imread(str(tp)))[:nfirst]
        sig_binned = bin_xy_sum(signal, bin)

        mean, real, imag = phasor_from_signal(sig_binned, axis=0)

        # ---- Calibration (coumarin reference)
        real, imag = phasor_calibrate(
            real, imag,
            ref_mean, ref_real, ref_imag,
            frequency=frequency_mhz,
            lifetime=ref_lifetime_ns,
        )

        # ---- Optional phasor median filtering (important for stable phase coloring)
        if do_filter:
            mean, real, imag = phasor_filter_median(
                mean, real, imag, size=filter_size, repeat=filter_repeat
            )

        # ---- Pseudocolor (robust phase + robust brightness)
        rgb = make_phase_pseudocolor(
            mean, real, imag,
            mean_min=mean_min,
            phase_cmap=phase_cmap,
            v_percentile=v_percentile,
            gamma=gamma,
            phase_percentiles=phase_percentiles,
        )
        pseudo_tiles.append(rgb)

        save_rgb(
            outdir / f"{tile_name}_pseudocolor.png",
            rgb,
            title=f"{tile_name} phase pseudocolor (bin={bin}, first={nfirst})",
        )

    # -----------------------
    # Assemble mosaic (snake / zig-zag)
    #   Row 1:  1  2  3  4
    #   Row 2:  8  7  6  5  (reversed)
    #   Row 3:  9 10 11 12
    #   Row 4: 16 15 14 13  (reversed)
    # -----------------------
    if len(pseudo_tiles) >= nrows * ncols:
        h0, w0, _ = pseudo_tiles[0].shape
        pseudo_tiles = [t[:h0, :w0, :] for t in pseudo_tiles]

        rows = []
        idx = 0
        for rr in range(nrows):
            row_tiles = pseudo_tiles[idx: idx + ncols]
            if snake_order and (rr % 2 == 1):
                row_tiles = row_tiles[::-1]
            rows.append(np.concatenate(row_tiles, axis=1))
            idx += ncols

        mosaic_rgb = np.concatenate(rows, axis=0)

        save_rgb(
            outdir / f"mosaic_{nrows}x{ncols}_pseudocolor.png",
            mosaic_rgb,
            title=f"Mosaic {nrows}x{ncols} pseudocolor (snake order)",
        )
        print(f"Saved: {outdir / f'mosaic_{nrows}x{ncols}_pseudocolor.png'}")
    else:
        print("Not enough tiles to assemble full pseudocolor mosaic.")

    print(f"Done. Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()