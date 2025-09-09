# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "waveorder",
#     "napari[pyqt6]",
#     "pdbpp",
#     "torch",
#     "tifffile",
# ]
# ///

from pathlib import Path

import napari
import tifffile as tf
import torch
from waveorder.models import phase_thick_3d

PARAMS = {
    "zyx_shape": (100, 256, 256),  # overwritten when data loads
    "yx_pixel_size": 6.5 / 60,
    "z_pixel_size": 0.5,
    "index_of_refraction_media": 1.520,
    "z_padding": 0,
    "wavelength_illumination": 0.525,
    "numerical_aperture_illumination": 0.52,
    "numerical_aperture_detection": 1.4,
}


def get_data():
    # download example data and cache it in ~/.cache/waveorder if not present...
    img_url = "https://www.dropbox.com/scl/fi/lykhkfkbrm65naxmjayup/finalstack.tif?rlkey=l6sjbj728ipdalhingc9x019m&dl=1"
    img_path = "finalstack.tif"
    cache_path = Path.home() / ".cache" / "waveorder" / img_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        import urllib.request

        print(f"Downloading {img_url} to {cache_path}...")
        urllib.request.urlretrieve(img_url, cache_path)
        print("Download complete.")
    else:
        print("Using cached image at:", cache_path)

    with tf.TiffFile(cache_path) as tif:
        return tif.asarray()


# read cached tiff into torch.Tensor
zyx_data = torch.tensor(get_data(), dtype=torch.float32)
PARAMS["zyx_shape"] = zyx_data.shape

# Calculate transfer function
otf = phase_thick_3d.calculate_transfer_function(**PARAMS)

# Reconstruct
zyx_recon = phase_thick_3d.apply_inverse_transfer_function(
    zyx_data, *otf, PARAMS["z_padding"]
)

# Display
viewer = napari.Viewer()
viewer.add_image(zyx_data.numpy(), name="Data")
viewer.add_image(zyx_recon.numpy(), name="Reconstruction")
viewer.grid.enabled = True
napari.run()
