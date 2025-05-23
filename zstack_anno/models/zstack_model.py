import os
import tifffile
import numpy as np
from ..utils.morphology_tools import (
    label_components,
    histogram_stretch_stack,
    remove_mask_background,
)

class ZStackModel:
    """Model holding the currently loaded Z-stack image and masks."""

    def __init__(self):
        self.data: np.ndarray | None = None
        self.original_data: np.ndarray | None = None
        self.masks: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.index: int = 0
        self.path: str | None = None
        self.mask_path: str | None = None
        self.mask_dirty: bool = False
        self.ome_metadata: str | None = None

    def load(self, path: str) -> None:
        """Load a TIFF stack and reset masks."""
        with tifffile.TiffFile(path) as tif:
            arr = tif.asarray()
            self.ome_metadata = tif.ome_metadata
        print("Loaded shape:", arr.shape, "dtype:", arr.dtype)

        # Remove single-length axes (e.g. t=1, c=1)
        arr = np.squeeze(arr)
        # If still 4-D, assume first axis is channel/time and take first slice
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError("Only 3-D stacks are supported")

        self.data = arr
        self.original_data = arr.copy()
        self.index = 0
        self.masks = None
        self.components = None
        self.path = path
        self.mask_path = None
        self.mask_dirty = False

    def load_masks(self, path: str) -> None:
        """Load mask stack from a TIFF file."""
        self.masks = tifffile.imread(path)
        self.mask_path = path
        self.mask_dirty = False
        self.update_components()

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

    def set_mask(
        self,
        mask: np.ndarray,
        slice_idx: int | None = None,
        update_components: bool = True,
    ) -> None:
        """Set mask for a slice. Creates mask stack if absent."""
        if self.data is None:
            raise RuntimeError("Image must be loaded before setting masks")
        if self.masks is None:
            self.masks = np.zeros_like(self.data, dtype=mask.dtype)
        if slice_idx is None:
            slice_idx = self.index
        self.masks[slice_idx] = mask
        self.mask_dirty = True
        if update_components:
            # lazily allocate components and update only the modified slice
            if self.components is None:
                self.components = np.zeros_like(self.masks, dtype=np.int32)
            if self.components.shape != self.masks.shape:
                self.components = np.zeros_like(self.masks, dtype=np.int32)
            self.components[slice_idx] = label_components(mask)

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
        self.components = np.zeros_like(self.masks, dtype=np.int32)

    def ensure_masks(self) -> None:
        """Ensure that an in-memory mask stack exists without writing to disk."""
        if self.data is None:
            raise RuntimeError("Image must be loaded before creating masks")
        if self.masks is None:
            self.masks = np.zeros_like(self.data, dtype=np.uint8)
            self.components = np.zeros_like(self.masks, dtype=np.int32)
            self.mask_dirty = False

    def update_components(self) -> None:
        """Recompute connected component labels for all masks."""
        if self.masks is None:
            self.components = None
            return
        labeled = [label_components(slice_) for slice_ in self.masks]
        self.components = np.stack(labeled)



    # 便利属性
    @property
    def n_slices(self) -> int:
        return 0 if self.data is None else self.data.shape[0]

    def get_current(self) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("No image loaded")
        return self.data[self.index]

    def total_pixel_count(self) -> int:
        """Return number of foreground pixels across all masks."""
        if self.masks is None:
            return 0
        return int(self.masks.sum())

    def component_count(self) -> int:
        """Return total connected component count across the stack."""
        if self.masks is None:
            return 0
        if self.components is None:
            self.update_components()
        return int(sum(self.components[i].max() for i in range(self.components.shape[0])))

    # --------- image utilities ---------
    def histogram_stretch(self, percentile: float) -> None:
        """Apply histogram stretch to the entire stack."""
        if self.original_data is None:
            raise RuntimeError("Image not loaded")
        self.data = histogram_stretch_stack(self.original_data, percentile)

    def reset_contrast(self) -> None:
        """Revert ``data`` to the original loaded image."""
        if self.original_data is not None:
            self.data = self.original_data.copy()

    def remove_background(self, percentile: float, slice_idx: int | None = None) -> None:
        """Remove low intensity pixels from the mask on ``slice_idx``."""
        if self.data is None or self.masks is None:
            return
        if slice_idx is None:
            slice_idx = self.index
        img = self.data[slice_idx]
        mask = self.masks[slice_idx]
        new_mask = remove_mask_background(img, mask, percentile)
        self.set_mask(new_mask, slice_idx)

    # --------- utility methods ---------
    def delete_components_touching_rect(
        self, slice_idx: int, x0: int, y0: int, x1: int, y1: int
    ) -> None:
        """Delete components that have any pixel within ``(x0,y0,x1,y1)``."""
        if self.masks is None:
            return
        if self.components is None:
            self.update_components()
        mask = self.masks[slice_idx]
        labels = self.components[slice_idx]
        sub = labels[y0:y1, x0:x1]
        to_del = np.unique(sub)
        to_del = to_del[to_del > 0]
        if to_del.size == 0:
            return
        new_mask = mask.copy()
        for lbl in to_del:
            new_mask[labels == lbl] = 0
        self.set_mask(new_mask, slice_idx)

    def truncate(self, start: int, end: int) -> None:
        """Truncate the stack to ``start``-``end`` inclusive and note in OME."""
        if self.data is None:
            return
        self.data = self.data[start : end + 1]
        if self.original_data is not None:
            self.original_data = self.original_data[start : end + 1]
        if self.masks is not None:
            self.masks = self.masks[start : end + 1]
        if self.components is not None:
            self.components = self.components[start : end + 1]
        self.index = 0
        if self.ome_metadata:
            note = f"Truncated from slice {start} to {end}"
            self.ome_metadata += f"\n<!-- {note} -->\n"

    def save_stack(self, path: str) -> None:
        """Save current image stack with OME metadata if available."""
        if self.data is None:
            raise RuntimeError("No image loaded")
        tifffile.imwrite(path, self.data, ome=self.ome_metadata)
