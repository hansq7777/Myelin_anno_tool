import os
import tifffile
import numpy as np

class ZStackModel:
    """Model holding the currently loaded Z-stack image and masks."""

    def __init__(self):
        self.data: np.ndarray | None = None
        self.masks: np.ndarray | None = None
        self.index: int = 0
        self.path: str | None = None
        self.mask_path: str | None = None
        self.mask_dirty: bool = False

    def load(self, path: str) -> None:
        """Load a TIFF stack and reset masks."""
        arr = tifffile.imread(path)
        print("Loaded shape:", arr.shape, "dtype:", arr.dtype)

        # Remove single-length axes (e.g. t=1, c=1)
        arr = np.squeeze(arr)
        # If still 4-D, assume first axis is channel/time and take first slice
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError("Only 3-D stacks are supported")

        self.data = arr
        self.index = 0
        self.masks = None
        self.path = path
        self.mask_path = None
        self.mask_dirty = False

    def load_masks(self, path: str) -> None:
        """Load mask stack from a TIFF file."""
        self.masks = tifffile.imread(path)
        self.mask_path = path
        self.mask_dirty = False

    def save_masks(self, path: str | None = None) -> None:
        """Save current mask stack as a TIFF file."""
        if self.masks is None:
            raise RuntimeError("No masks to save")
        if path is None:
            if self.mask_path is None:
                raise RuntimeError("No path specified for saving masks")
            path = self.mask_path
        tifffile.imwrite(path, self.masks.astype(np.uint8))
        self.mask_path = path
        self.mask_dirty = False

    def get_mask(self, slice_idx: int | None = None) -> np.ndarray:
        """Return mask array for the given slice (defaults to current)."""
        if self.masks is None:
            raise RuntimeError("No masks loaded")
        if slice_idx is None:
            slice_idx = self.index
        return self.masks[slice_idx]

    def set_mask(self, mask: np.ndarray, slice_idx: int | None = None) -> None:
        """Set mask for a slice. Creates mask stack if absent."""
        if self.data is None:
            raise RuntimeError("Image must be loaded before setting masks")
        if self.masks is None:
            self.masks = np.zeros_like(self.data, dtype=mask.dtype)
        if slice_idx is None:
            slice_idx = self.index
        self.masks[slice_idx] = mask
        self.mask_dirty = True

    # --------- mask helpers ---------
    def default_mask_path(self) -> str:
        if self.path is None:
            raise RuntimeError("Image not loaded")
        base = os.path.splitext(os.path.basename(self.path))[0] + "_mask.tif"
        return os.path.join(os.path.dirname(self.path), base)

    def create_blank_masks(self, path: str | None = None) -> None:
        if self.data is None:
            raise RuntimeError("Image must be loaded before creating masks")
        if path is None:
            path = self.default_mask_path()
        self.masks = np.zeros_like(self.data, dtype=np.uint8)
        self.mask_path = path
        tifffile.imwrite(self.mask_path, self.masks)
        self.mask_dirty = False



    # 便利属性
    @property
    def n_slices(self) -> int:
        return 0 if self.data is None else self.data.shape[0]

    def get_current(self) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("No image loaded")
        return self.data[self.index]
