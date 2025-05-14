# Placeholder for OWLv2 data loaders (if needed)

from PIL import Image
import numpy as np
import torch

def load_image_any_format(path: str, mode: str = "RGB"):
    """
    Load an image from path, supporting TIFF and N-channel images if possible.
    Args:
        path: Path to image file
        mode: Color mode (e.g., 'RGB', 'I;16', etc.)
    Returns:
        np.ndarray or torch.Tensor
    """
    try:
        img = Image.open(path)
        if mode:
            img = img.convert(mode)
        arr = np.array(img)
        return arr
    except Exception as e:
        print(f"[OWLv2] Failed to load image {path}: {e}")
        return None

def load_image_n_channel(path: str):
    """
    Load an image as a numpy array, preserving all channels (for N-channel TIFFs).
    Returns:
        np.ndarray or None
    """
    try:
        img = Image.open(path)
        arr = np.array(img)
        # If the image is a multi-page TIFF, stack all pages
        if hasattr(img, "n_frames") and img.n_frames > 1:
            arr = np.stack([np.array(img.seek(i) or img) for i in range(img.n_frames)])
        return arr
    except Exception as e:
        print(f"[OWLv2] Failed to load N-channel image {path}: {e}")
        return None

def load_image_high_bitdepth(path: str):
    """
    Load a high bit-depth image (e.g., 16-bit TIFF) as a numpy array.
    Returns:
        np.ndarray or None
    """
    try:
        img = Image.open(path)
        arr = np.array(img)
        # If dtype is not uint8, print info
        if arr.dtype != np.uint8:
            print(f"[OWLv2] Loaded high bit-depth image {path} with dtype {arr.dtype}")
        return arr
    except Exception as e:
        print(f"[OWLv2] Failed to load high bit-depth image {path}: {e}")
        return None

# Note: Pillow may not support all N-channel/high bit-depth TIFFs. For advanced use, consider tifffile or rasterio.
# Add more loader utilities as needed for N-channel/high bit-depth TIFFs
# Document limitations if Pillow cannot handle certain formats
