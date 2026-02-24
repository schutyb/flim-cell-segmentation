from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage import exposure

from phasorpy.phasor import phasor_from_signal
from phasorpy.lifetime import phasor_calibrate
from phasorpy.plot import plot_phasor_image, plot_phasor


# -----------------------
# Paths + params
# -----------------------
path = Path(""
"/Users/schutyb/Documents/cell_segmentation/dod/p427/visit01/Mosaic02_4x4_FOV600_z120_32A1/Im_00004.tif")
path_ref = Path(""
"/Users/schutyb/Documents/cell_segmentation/dod/p427/visit01/Image01_FOV600_z210_32A1/Im_00001.tif")

bin = 4
frequency_mhz = 80
ref_lifetime_ns = 2.5

do_filter = True

# -----------------------
# Load
# -----------------------
signal = tifffile.imread(str(path))          # expected (H, Y, X)
ref_signal = tifffile.imread(str(path_ref))  # expected (H, Y, X)


# -----------------------
# Binning (sum photons) on spatial axes only
# -----------------------
H, Y, X = signal.shape
Y2 = (Y // bin) * bin
X2 = (X // bin) * bin
sig = signal[:, :Y2, :X2]
sig_binned = sig.reshape(
    H, Y2 // bin, bin, X2 // bin, bin).sum(axis=(2, 4))

# Apply SAME binning to reference for consistency
Hr, Yr, Xr = ref_signal.shape
Yr2 = (Yr // bin) * bin
Xr2 = (Xr // bin) * bin
ref = ref_signal[:, :Yr2, :Xr2]
ref_binned = ref.reshape(
    Hr, Yr2 // bin, bin, Xr2 // bin, bin).sum(axis=(2, 4))


# -----------------------
# Phasor
# -----------------------
mean, real, imag = phasor_from_signal(sig_binned, axis=0)
ref_mean, ref_real, ref_imag = phasor_from_signal(ref_binned, axis=0)


# -----------------------
# Calibration
# -----------------------
real, imag = phasor_calibrate(
    real, imag,
    ref_mean, ref_real, ref_imag,
    frequency=frequency_mhz,
    lifetime=ref_lifetime_ns,
)


# -----------------------
# Optional filtering (on phasor images)
# -----------------------
if do_filter:
    from phasorpy.filter import phasor_filter_median
    mean, real, imag = phasor_filter_median(
        mean, real, imag, size=3, repeat=2)



# -----------------------
# Phasor image + standard phasor plot (phasorpy)
# -----------------------
plot_phasor_image(mean, real, imag, title="Calibrated (binned)")

plot_phasor(
    real, imag,
    frequency=frequency_mhz,
    cmap="RdYlGn_r",
    title="Calibrated phasor coordinates",
)

plt.close('all')

# COLOR MAP ---------------------------------------------------------
plotty = False
if plotty:
    import matplotlib.cm as cm

    # ---- Params
    mean_min = 5  # threshold para ignorar fondo/ruido

    # ---- Mask por intensidad
    mask = mean > mean_min

    # ---- Phase (atan2) y normalización a [0,1]
    phase = np.arctan2(imag, real)  # rad
    pmin = np.nanmin(phase[mask])
    pmax = np.nanmax(phase[mask])
    phase_norm = (phase - pmin) / (pmax - pmin + 1e-12)
    phase_norm = np.clip(phase_norm, 0, 1)
    phase_norm[~mask] = 0.0

    # ---- Colormap espectral (violeta->azul->verde->amarillo->naranja->rojo)
    cmap = cm.get_cmap("nipy_spectral")  # probá también "turbo"
    rgb = cmap(phase_norm)[..., :3]      # RGBA -> RGB

    # ---- Brillo = intensidad (normalizada)
    val = mean.astype(np.float32)
    vmax = np.nanmax(val)
    val = val / vmax if vmax > 0 else val
    val[~mask] = 0.0

    # Modulación por brillo
    rgb = rgb * val[..., None]

    # ---- Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(f"Phasor phase spectral (brightness=intensity), mean>{mean_min}")
    plt.axis("off")
    plt.show()

# --------------- Clustering ---------------------------------------------------------
cluster = True
if cluster:
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap
    from sklearn.cluster import KMeans

    # -----------------------
    # Params
    # -----------------------
    mean_min = 5
    n_clusters = 5

    # 6 colores: violeta -> azul -> verde -> amarillo -> naranja -> rojo
    spectral_colors = ["#6A0DAD", "#0000FF", "#00FF00", "#FFFF00", "#FFA500", "#FF0000"]

    # -----------------------
    # Custom colormaps
    # -----------------------
    phase_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_spectral", spectral_colors, N=256
    )
    cluster_cmap = ListedColormap(spectral_colors[:n_clusters], name="cluster6")

    # -----------------------
    # Pseudocolor by phase (brightness = intensity)
    # -----------------------
    mask = np.isfinite(real) & np.isfinite(imag) & np.isfinite(mean) & (mean > mean_min)

    phase = np.arctan2(imag, real)
    pmin = np.nanmin(phase[mask])
    pmax = np.nanmax(phase[mask])
    phase_norm = (phase - pmin) / (pmax - pmin + 1e-12)
    phase_norm = np.clip(phase_norm, 0, 1)
    phase_norm[~mask] = 0.0

    rgb = phase_cmap(phase_norm)[..., :3]

    val = mean.astype(np.float32)
    vmax = np.nanmax(val)
    val = val / vmax if vmax > 0 else val
    val[~mask] = 0.0

    rgb *= val[..., None]

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(f"Phase pseudocolor (custom spectral), brightness=intensity, mean>{mean_min}")
    plt.axis("off")

    # -----------------------
    # KMeans clustering on (G,S) with mask
    # -----------------------
    g = real.ravel()
    s = imag.ravel()
    m = mean.ravel()

    valid = np.isfinite(g) & np.isfinite(s) & np.isfinite(m) & (m > mean_min)
    features = np.stack([g[valid], s[valid]], axis=1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels_valid = kmeans.fit_predict(features)

    # Build label image (background = -1)
    label_image = np.full(real.shape, -1, dtype=np.int16)
    label_image.ravel()[valid] = labels_valid

    # -----------------------
    # OPTIONAL: order clusters by mean phase so colors follow spectral order
    # (violet -> ... -> red)
    # -----------------------
    phase_flat = phase.ravel()[valid]
    cluster_phase_mean = np.array([np.nanmean(phase_flat[labels_valid == k]) for k in range(n_clusters)])
    order = np.argsort(cluster_phase_mean)

    remap = np.empty_like(order)
    remap[order] = np.arange(n_clusters)  # old -> new
    labels_valid_ord = remap[labels_valid]
    label_image_ord = np.full(real.shape, -1, dtype=np.int16)
    label_image_ord.ravel()[valid] = labels_valid_ord

    # -----------------------
    # Plot cluster image (custom colors)
    # -----------------------
    plt.figure(figsize=(6, 6))
    im = plt.imshow(label_image_ord, cmap=cluster_cmap, vmin=0, vmax=n_clusters - 1)
    plt.title(f"KMeans clusters (n={n_clusters}, ordered by phase), mean>{mean_min}")
    plt.axis("off")
    plt.colorbar(im, label="Cluster (spectral order)")

    # -----------------------
    # Phasor scatter colored by ordered clusters + semicircle
    # -----------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels_valid_ord, s=1, cmap=cluster_cmap, alpha=0.6)
    plt.xlabel("G (real)")
    plt.ylabel("S (imag)")
    plt.title("Phasor scatter colored by KMeans cluster (spectral colors)")
    plt.gca().set_aspect("equal")

    t = np.linspace(0, 1, 400)
    plt.plot(t, np.sqrt(np.clip(t - t**2, 0, None)), color="black", linewidth=1)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()