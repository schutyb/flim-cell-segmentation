#!/usr/bin/env python3
"""
FLIM -> Intensity + Phase Pseudocolor (calibrated; matches previous look)

Reads FLIM TIFF tiles (time bins x Y x X), computes:
  - intensity image (phasor mean, equivalent to summed/mean photon counts per pixel),
  - calibrated phasor (g,s) using a coumarin reference,
  - pseudocolor RGB where:
      * hue encodes phasor phase atan2(s,g) with robust percentile normalization,
      * brightness encodes intensity with robust percentile normalization + gamma,
      * background is masked using mean_min.

Saves per tile:
  - *_intensity.tif
  - *_pseudocolor.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile
import imageio.v2 as imageio

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors

from phasorpy.phasor import phasor_from_signal
from phasorpy.lifetime import phasor_calibrate


# -----------------------
# Helpers (same logic as your previous script)
# -----------------------
def bin_xy_sum(signal: np.ndarray, bin: int) -> np.ndarray:
    """Spatial binning by summing photons in (Y,X). Expects (H, Y, X)."""
    H, Y, X = signal.shape
    Y2 = (Y // bin) * bin
    X2 = (X // bin) * bin
    sig = signal[:, :Y2, :X2]
    return sig.reshape(H, Y2 // bin, bin, X2 // bin, bin).sum(axis=(2, 4))


def ensure_time_first(arr: np.ndarray) -> np.ndarray:
    """
    Ensure FLIM array is (H, Y, X) where H = time bins.
    Common cases: already (H,Y,X) or (Y,X,H).
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D FLIM (H,Y,X). Got shape={arr.shape}")

    shape = arr.shape
    # time bins often between 16..2048 and usually the smallest dimension
    candidates = [ax for ax, n in enumerate(shape) if 16 <= n <= 2048]
    h_ax = min(candidates, key=lambda ax: shape[ax]) if candidates else int(np.argmin(shape))

    if h_ax != 0:
        arr = np.moveaxis(arr, h_ax, 0)
    return arr


def make_phase_pseudocolor(
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    mean_min: float,
    phase_cmap: mcolors.Colormap,
    v_percentile: float = 99.0,
    gamma: float = 0.8,
    phase_percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """
    RGB where:
      - color = phase = atan2(imag, real), robust-normalized with percentiles
      - brightness = mean intensity, robust-normalized with percentile + gamma
    Returns uint8 RGB.
    """
    mean_f = mean.astype(np.float32)
    mask = np.isfinite(real) & np.isfinite(imag) & np.isfinite(mean_f) & (mean_f > mean_min)

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

    # ---- Brightness
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
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# -----------------------
# Main
# -----------------------
def main() -> None:
    # -----------------------
    # EDIT THESE PATHS
    # -----------------------
    folder = Path(
        "/Users/schutyb/Documents/cell_segmentation/dod/data_raw/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp")
    
    ref_path = Path(
        "/Users/schutyb/Documents/cell_segmentation/dod/data_raw/p449/visit01/coumarin6_ExVivo_slide/Image03_FOV256_z152_32Sp/Im_00001.tif")

    outdir = Path(
        "/Users/schutyb/Documents/cell_segmentation/dod/data_process/p449/visit01")
    
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # PARAMS (match your previous good-looking output)
    # -----------------------
    bin_xy = 4
    nfirst = 16
    frequency_mhz = 80.0
    ref_lifetime_ns = 2.5

    mean_min = 3.0
    v_percentile = 99.0
    gamma = 0.95
    phase_percentiles = (0.0, 100.0)

    do_filter = True
    filter_size = 3
    filter_repeat = 2

    # Colormap: violet -> blue -> green -> yellow -> orange -> red
    spectral_colors = ["#6A0DAD", "#0000FF", "#00FF00", "#FFFF00", "#FFA500", "#FF0000"]
    phase_cmap = mcolors.LinearSegmentedColormap.from_list("custom_spectral", spectral_colors, N=256)

    # -----------------------
    # Load tiles
    # -----------------------
    tile_paths = sorted(folder.glob("Im_*.tif"))
    if not tile_paths:
        raise SystemExit(f"No Im_*.tif found in: {folder}")

    # -----------------------
    # Prepare reference once
    # -----------------------
    ref_signal = np.asarray(tifffile.imread(str(ref_path)))
    ref_signal = ensure_time_first(ref_signal)[:nfirst].astype(np.float32)
    ref_binned = bin_xy_sum(ref_signal, bin_xy)

    ref_mean, ref_real, ref_imag = phasor_from_signal(ref_binned, axis=0)

    if do_filter:
        from phasorpy.filter import phasor_filter_median

    # -----------------------
    # Process tiles
    # -----------------------
    for i, tp in enumerate(tile_paths, start=1):
        tile_name = tp.stem
        print(f"[{i}/{len(tile_paths)}] {tile_name}")

        signal = np.asarray(tifffile.imread(str(tp)))
        signal = ensure_time_first(signal)[:nfirst].astype(np.float32)
        sig_binned = bin_xy_sum(signal, bin_xy)

        mean, real, imag = phasor_from_signal(sig_binned, axis=0)

        # Calibration (coumarin reference)
        real, imag = phasor_calibrate(
            real, imag,
            ref_mean, ref_real, ref_imag,
            frequency=frequency_mhz,
            lifetime=ref_lifetime_ns,
        )

        # Optional median filtering (stabilizes phase coloring)
        if do_filter:
            mean, real, imag = phasor_filter_median(
                mean, real, imag, size=filter_size, repeat=filter_repeat
            )

        # Save intensity (use mean from phasorpy; consistent with your previous pipeline)
        intensity = mean.astype(np.float32)
        tifffile.imwrite(outdir / f"{tile_name}_intensity.tif", intensity)

        # Save pseudocolor (matches previous make_phase_pseudocolor logic)
        rgb8 = make_phase_pseudocolor(
            mean, real, imag,
            mean_min=mean_min,
            phase_cmap=phase_cmap,
            v_percentile=v_percentile,
            gamma=gamma,
            phase_percentiles=phase_percentiles,
        )
        imageio.imwrite(outdir / f"{tile_name}_pseudocolor.png", rgb8)

    print(f"Done. Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
