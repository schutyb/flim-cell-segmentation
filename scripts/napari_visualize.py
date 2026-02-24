import napari
import tifffile
from skimage.io import imread

# --- Paths (ajustá)
I_path = "/Users/schutyb/Documents/cell_segmentation/dod/data_process/p449/visit01/Im_00001_intensity.tif"
L_path = "/Users/schutyb/Documents/cell_segmentation/dod/data_process/p449/visit01/Im_00001_pseudocolor_instances.tif"  # tu máscara de instancias

# --- Load
I = imread(I_path)            # (Y,X) o (Z,Y,X)
labels = tifffile.imread(L_path)

viewer = napari.Viewer()

# Intensity base
viewer.add_image(I, name="Intensity", colormap="gray", contrast_limits=None)

# Labels overlay
lab = viewer.add_labels(labels, name="Instances", opacity=0.55)

# Mostrar TODAS las instancias con colores distintos
lab.color_mode = "auto"
lab.show_selected_label = False

# --- Elegí una de estas dos opciones:
# 1) Relleno completo (más vistoso)
lab.contour = 0

# 2) Solo contorno (mejor para QA)
# lab.contour = 2
# lab.opacity = 1.0

viewer.reset_view()
napari.run()