import napari
import numpy as np
from skimage.io import imread
from pathlib import Path
import tifffile

# -------- Path a tu pseudocolor
P_path = Path("/Users/schutyb/Documents/cell_segmentation/dod/data_process/p449/visit01/Im_00001_pseudocolor.png")

# Guarda la binaria al lado del pseudocolor
M_path = P_path.with_name(P_path.stem + "_mask.tif")  # 0=fondo, 1=célula

# -------- Load pseudocolor
P = imread(str(P_path))

viewer = napari.Viewer()
viewer.add_image(P, name="Pseudocolor", rgb=True)

# -------- Create/load binary mask (0/1)
mask_shape = P.shape[:2] if P.ndim == 3 else P.shape[:-1]  # (Y,X) or (Z,Y,X)

if M_path.exists():
    mask = tifffile.imread(str(M_path))
    # convert anything nonzero -> 1
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != mask_shape:
        raise ValueError(f"Existing mask shape {mask.shape} doesn't match image shape {mask_shape}")
    print(f"Loaded existing mask: {M_path.name}")
else:
    mask = np.zeros(mask_shape, dtype=np.uint8)

lab = viewer.add_labels(mask, name="CellMask", opacity=0.6)

# --- Make it easy: always paint label 1
lab.selected_label = 1
lab.show_selected_label = False
lab.color_mode = "auto"

# Optional: show only contours while editing (less intrusive)
# lab.contour = 2
# lab.opacity = 1.0

# -------- Autosave on close (Qt-level, compatible)
qt_win = viewer.window._qt_window
old_close_event = qt_win.closeEvent

def new_close_event(event):
    try:
        # save strictly 0/1
        out = (lab.data > 0).astype(np.uint8)
        tifffile.imwrite(str(M_path), out)
        print(f"[Saved] {M_path.resolve()}")
    finally:
        old_close_event(event)

qt_win.closeEvent = new_close_event

napari.run()

