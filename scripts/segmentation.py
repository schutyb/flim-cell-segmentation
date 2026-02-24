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
from skimage.morphology import white_tophat, disk
from skimage.filters import threshold_otsu

from sklearn.mixture import GaussianMixture

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
def make_phase_pseudocolor_like_before(
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    mean_min: float,
    phase_cmap: mcolors.Colormap,
    v_percentile: float = 99.5,
    gamma: float = 0.9,
    phase_percentiles: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    "Like before": per-tile robust stretch of PHASE to fill the full colormap,
    plus brightness from mean.

    - color: phase stretched by percentiles (per tile)
    - brightness: mean normalized by percentile + gamma
    """
    mask = np.isfinite(real) & np.isfinite(imag) & np.isfinite(mean) & (mean > mean_min)
    phase = np.arctan2(imag, real)

    # phase robust stretch per tile (avoid everything yellow)
    if np.any(mask):
        p_lo, p_hi = np.nanpercentile(phase[mask], phase_percentiles)
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or (p_hi - p_lo) < 1e-6:
            p_lo, p_hi = -np.pi, np.pi
    else:
        p_lo, p_hi = -np.pi, np.pi

    phase_norm = np.clip((phase - p_lo) / (p_hi - p_lo + 1e-12), 0, 1)
    phase_norm[~mask] = 0.0
    rgb = phase_cmap(phase_norm)[..., :3].astype(np.float32)

    # brightness from mean (robust percentile)
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


def overlay_boundaries_rgb(base_rgb: np.ndarray, binary_mask: np.ndarray, color=(1.0, 1.0, 1.0)) -> np.ndarray:
    b = segmentation.find_boundaries(binary_mask, mode="outer")
    out = base_rgb.copy()
    out[b] = color
    return out


def overlay_on_grayscale(mean: np.ndarray, binary_mask: np.ndarray, color=(1.0, 0.0, 0.0)) -> np.ndarray:
    """
    Mean image in grayscale with colored boundaries on top.
    Returns RGB.
    """
    mean_f = mean.astype(np.float32)
    finite = np.isfinite(mean_f)
    if np.any(finite):
        vmin, vmax = np.nanpercentile(mean_f[finite], [1, 99])
    else:
        vmin, vmax = 0.0, 1.0
    mean_norm = np.clip((mean_f - vmin) / (vmax - vmin + 1e-12), 0, 1)

    rgb = np.stack([mean_norm, mean_norm, mean_norm], axis=-1).astype(np.float32)
    b = segmentation.find_boundaries(binary_mask, mode="outer")
    rgb[b] = color
    return np.clip(rgb, 0, 1)


# -----------------------
# Two-stage segmentation
# -----------------------
def detect_candidates_by_intensity(
    mean: np.ndarray,
    mean_min: float,
    tophat_radius: int = 2,
    min_area: int = 4,
    max_area: int = 120,
) -> np.ndarray:
    """
    Candidate detection using intensity only:
      - white top-hat to enhance small bright blobs
      - Otsu threshold on enhanced image (valid pixels)
      - size filtering
    Returns labeled candidates.
    """
    mean_f = mean.astype(np.float32)
    valid = np.isfinite(mean_f) & (mean_f > mean_min)
    if not np.any(valid):
        return np.zeros_like(mean, dtype=np.int32)

    enh = white_tophat(mean_f, disk(tophat_radius))
    v = enh[valid]
    thr = threshold_otsu(v) if v.size else 0.0

    cand = valid & (enh > thr)
    cand = morphology.remove_small_objects(cand, min_size=min_area)

    labels = measure.label(cand)
    if labels.max() == 0:
        return labels.astype(np.int32)

    props = measure.regionprops(labels)
    rm = [p.label for p in props if p.area > max_area]
    if rm:
        labels[np.isin(labels, rm)] = 0
        labels = measure.label(labels > 0)

    return labels.astype(np.int32)


def classify_candidates_by_phasor(
    labels_cand: np.ndarray,
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    mean_min: float,
    n_components: int = 2,
    feature_mode: str = "gs",     # "gs" (G,S) suele ser más estable
    pick_rule: str = "min_area",  # cells suelen ser más chicas
) -> tuple[np.ndarray, dict]:
    """
    Classify candidate objects using phasor features.
    Returns binary cells mask (bool) and info.
    """
    if labels_cand.max() == 0:
        return np.zeros_like(labels_cand, dtype=bool), {"n_candidates": 0, "n_cells": 0}

    phase = np.arctan2(imag, real)
    mod = np.sqrt(real**2 + imag**2)

    mean_f = mean.astype(np.float32)
    props = measure.regionprops(labels_cand, intensity_image=mean_f)

    feats = []
    meta = []  # (label, area, phase_mean)
    for p in props:
        coords = p.coords
        rr = coords[:, 0]
        cc = coords[:, 1]

        ok = (
            np.isfinite(mean_f[rr, cc]) &
            np.isfinite(real[rr, cc]) &
            np.isfinite(imag[rr, cc]) &
            (mean_f[rr, cc] > mean_min)
        )
        if ok.sum() < 2:
            continue

        rr = rr[ok]
        cc = cc[ok]

        phm = float(np.nanmean(phase[rr, cc]))
        mdm = float(np.nanmean(mod[rr, cc]))
        gm = float(np.nanmean(real[rr, cc]))
        sm = float(np.nanmean(imag[rr, cc]))

        if feature_mode == "phase_mod":
            feats.append([phm, mdm])
        else:
            feats.append([gm, sm])

        meta.append((p.label, p.area, phm))

    if len(feats) < max(6, n_components * 3):
        return np.zeros_like(labels_cand, dtype=bool), {
            "n_candidates": int(labels_cand.max()),
            "n_cells": 0,
            "note": "too_few_objects"
        }

    X = np.asarray(feats, dtype=np.float32)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    y = gmm.fit_predict(X)

    # pick "cell cluster"
    if pick_rule == "min_phase":
        phases = np.array([m[2] for m in meta], dtype=np.float32)
        means = [phases[y == k].mean() if np.any(y == k) else np.inf for k in range(n_components)]
        cell_cluster = int(np.argmin(means))
    else:
        areas = np.array([m[1] for m in meta], dtype=np.float32)
        means = [areas[y == k].mean() if np.any(y == k) else np.inf for k in range(n_components)]
        cell_cluster = int(np.argmin(means))

    cells_mask = np.zeros_like(labels_cand, dtype=bool)
    kept = 0
    for (lab, area, phm), cls in zip(meta, y):
        if cls == cell_cluster:
            cells_mask |= (labels_cand == lab)
            kept += 1

    return cells_mask, {
        "n_candidates": int(labels_cand.max()),
        "n_cells": int(kept),
        "feature_mode": feature_mode,
        "pick_rule": pick_rule,
        "cell_cluster": int(cell_cluster),
    }


def main():
    # -----------------------
    # Paths
    # -----------------------
    folder = Path("/Users/schutyb/Documents/cell_segmentation/dod/data_raw/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp")
    ref_path = Path(
        "/Users/schutyb/Documents/cell_segmentation/dod/data_raw/p449/visit01/coumarin6_ExVivo_slide"
        "/Image03_FOV256_z152_32Sp/Im_00001.tif"
    )

    outdir = Path("/Users/schutyb/Documents/cell_segmentation/dod/data_process/p449/visit01/two_stage_cells_full")
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
    phase_percentiles = (2.0, 98.0)

    spectral_colors = ["#6A0DAD", "#0000FF", "#00FF00", "#FFFF00", "#FFA500", "#FF0000"]
    phase_cmap = mcolors.LinearSegmentedColormap.from_list("custom_spectral", spectral_colors, N=256)

    # candidates
    tophat_radius = 2
    min_area = 4
    max_area = 120

    # classification
    n_components = 2
    feature_mode = "gs"
    pick_rule = "min_area"

    # mosaic
    nrows, ncols = 4, 4
    snake_order = True

    # -----------------------
    # Tiles
    # -----------------------
    tile_paths = sorted(folder.glob("Im_*.tif"), key=parse_tile_index)
    tile_paths = tile_paths[: nrows * ncols]
    if not tile_paths:
        raise SystemExit(f"No Im_*.tif found in {folder}")

    # -----------------------
    # Reference phasor
    # -----------------------
    ref_signal = load_tiff_hyx(ref_path)[:nfirst]
    ref_binned = bin_xy_sum(ref_signal, bin)
    ref_mean, ref_real, ref_imag = phasor_from_signal(ref_binned, axis=0)

    pseudo_tiles = []
    pseudo_overlay_tiles = []
    gray_overlay_tiles = []

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

        # calibrate
        real, imag = phasor_calibrate(
            real, imag,
            ref_mean, ref_real, ref_imag,
            frequency=frequency_mhz,
            lifetime=ref_lifetime_ns,
        )

        # filter
        if do_filter:
            mean, real, imag = phasor_filter_median(mean, real, imag, size=filter_size, repeat=filter_repeat)

        # pseudocolor tile (like before)
        rgb = make_phase_pseudocolor_like_before(
            mean, real, imag,
            mean_min=mean_min,
            phase_cmap=phase_cmap,
            v_percentile=v_percentile,
            gamma=gamma,
            phase_percentiles=phase_percentiles,
        )
        pseudo_tiles.append(rgb)
        plt.imsave(outdir / "tiles" / f"{tile_name}_pseudocolor.png", rgb)

        # candidates + classify
        labels_cand = detect_candidates_by_intensity(
            mean,
            mean_min=mean_min,
            tophat_radius=tophat_radius,
            min_area=min_area,
            max_area=max_area,
        )
        cells_mask, info = classify_candidates_by_phasor(
            labels_cand, mean, real, imag,
            mean_min=mean_min,
            n_components=n_components,
            feature_mode=feature_mode,
            pick_rule=pick_rule,
        )
        print(f"  candidates={info.get('n_candidates', 0)}  kept_cells={info.get('n_cells', 0)}")

        # overlays (pseudocolor + grayscale)
        ov_pseudo = overlay_boundaries_rgb(rgb, cells_mask, color=(1, 1, 1))
        ov_gray = overlay_on_grayscale(mean, cells_mask, color=(1, 0, 0))

        pseudo_overlay_tiles.append(ov_pseudo)
        gray_overlay_tiles.append(ov_gray)

        # save per-tile overlays
        plt.imsave(outdir / "tiles" / f"{tile_name}_cells_overlay_pseudocolor.png", ov_pseudo)
        plt.imsave(outdir / "tiles" / f"{tile_name}_cells_overlay_gray.png", ov_gray)

    # -----------------------
    # Mosaics
    # -----------------------
    if len(pseudo_tiles) >= nrows * ncols:
        mosaic_rgb = assemble_snake(pseudo_tiles[: nrows * ncols], nrows, ncols, snake_order=snake_order)
        plt.imsave(outdir / "mosaic" / "mosaic_pseudocolor.png", mosaic_rgb)

    if len(pseudo_overlay_tiles) >= nrows * ncols:
        mosaic_pseudo_ov = assemble_snake(pseudo_overlay_tiles[: nrows * ncols], nrows, ncols, snake_order=snake_order)
        plt.imsave(outdir / "mosaic" / "mosaic_cells_overlay_pseudocolor.png", mosaic_pseudo_ov)

    if len(gray_overlay_tiles) >= nrows * ncols:
        mosaic_gray_ov = assemble_snake(gray_overlay_tiles[: nrows * ncols], nrows, ncols, snake_order=snake_order)
        plt.imsave(outdir / "mosaic" / "mosaic_cells_overlay_gray.png", mosaic_gray_ov)

    print(f"Done. Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()