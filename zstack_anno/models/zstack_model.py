import os
import json
from typing import Callable
import tifffile
import numpy as np
from xml.etree import ElementTree as ET
from scipy.ndimage import zoom
from ..utils.morphology_tools import (
    label_components,
    histogram_stretch_stack,
    remove_mask_background,
    gaussian_blur_stack,
    threshold_absolute,
    threshold_normalized,
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
        self.blur_sigma: float = 0.0
        self.stretch_percent: float = 0.0
        self.reverse: bool = False
        self.show_original: bool = False
        # per-slice intensity stats and segmentation mask
        self._slice_intensity: np.ndarray | None = None
        self._seg_params: tuple[float, bool] | None = None
        self._segment_mask: np.ndarray | None = None
        self.mask_alignment_note: str | None = None
        self.image_revision: int = 0
        self.mask_revision: int = 0

    def _touch_image_revision(self) -> None:
        self.image_revision += 1

    def _touch_mask_revision(self) -> None:
        self.mask_revision += 1

    def load(self, path: str) -> None:
        """Load a stack from TIFF or CZI and reset masks."""
        if path.lower().endswith('.czi'):
            from ..utils.czi_utils import read_czi_stack, CziNotSupportedError

            try:
                arr, self.ome_metadata = read_czi_stack(path)
            except CziNotSupportedError as exc:
                raise RuntimeError(str(exc)) from exc
        else:
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

        # Keep a single copy of the loaded data until modifications are needed
        self.original_data = arr
        # ``data`` initially references the same array to avoid an immediate copy
        self.data = arr
        self.index = 0
        self.masks = None
        self.components = None
        self.path = path
        self.mask_path = None
        self.mask_dirty = False
        self.blur_sigma = 0.0
        self.stretch_percent = 0.0
        self.reverse = False
        self.show_original = False
        self._slice_intensity = None
        self._seg_params = None
        self._segment_mask = None
        self.mask_alignment_note = None
        self._touch_image_revision()
        self._touch_mask_revision()

    def load_masks(self, path: str) -> None:
        """Load mask stack from a TIFF file."""
        masks = tifffile.imread(path)
        masks = np.squeeze(masks)
        if masks.ndim == 4:
            masks = masks[0]
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        if masks.ndim != 3:
            raise ValueError(f"Mask stack must be 3-D after squeeze, got shape={masks.shape}")

        # Normalize to a binary uint8 stack for stable downstream ops.
        masks = (masks > 0).astype(np.uint8)

        self.mask_alignment_note = None
        if self.data is not None:
            dz, dy, dx = self.data.shape
            mz, my, mx = masks.shape
            if (my, mx) != (dy, dx):
                raise ValueError(
                    f"Mask XY shape {my}x{mx} does not match image shape {dy}x{dx}"
                )
            if mz != dz:
                masks = self._align_mask_depth(masks, dz)
                self.mask_alignment_note = f"Mask depth aligned {mz}->{dz} (nearest Z mapping)."

        self.masks = masks
        self.mask_path = path
        self.mask_dirty = False
        self.update_components()
        self._touch_mask_revision()

    def resample_image_to_shape(self, target_shape: tuple[int, int, int]) -> bool:
        """Resample loaded image stack in memory to ``target_shape`` (Z, Y, X).

        This is used by review mode when prediction/annotation stacks are stored
        at lower depth than raw stacks. Returned value indicates whether
        resampling was actually applied.
        """
        if self.original_data is None:
            raise RuntimeError("Image not loaded")
        if len(target_shape) != 3:
            raise ValueError(f"target_shape must be 3-D, got {target_shape}")
        target_z, target_y, target_x = (int(v) for v in target_shape)
        if target_z <= 0 or target_y <= 0 or target_x <= 0:
            raise ValueError(f"Invalid target_shape: {target_shape}")

        src = self.original_data
        src_shape = tuple(int(v) for v in src.shape)
        if src_shape == (target_z, target_y, target_x):
            return False

        zoom_factors = (
            target_z / src_shape[0],
            target_y / src_shape[1],
            target_x / src_shape[2],
        )
        src_float = src.astype(np.float32, copy=False)
        out = zoom(src_float, zoom_factors, order=1)

        if np.issubdtype(src.dtype, np.integer):
            info = np.iinfo(src.dtype)
            out = np.clip(np.rint(out), info.min, info.max).astype(src.dtype)
        else:
            out = out.astype(src.dtype, copy=False)

        self.original_data = out
        self.data = out.copy()
        self.index = min(self.index, out.shape[0] - 1)

        # Reset masks/components for the new grid; caller should load masks next.
        self.masks = None
        self.components = None
        self.mask_path = None
        self.mask_dirty = False
        self._slice_intensity = None
        self._seg_params = None
        self._segment_mask = None
        self._touch_image_revision()
        self._touch_mask_revision()

        # Keep physical extent by updating pixel size metadata if available.
        sizes = self.get_pixel_sizes()
        if sizes is not None:
            sx, sy, sz = sizes
            new_x = sx * src_shape[2] / target_x
            new_y = sy * src_shape[1] / target_y
            new_z = sz * src_shape[0] / target_z
            self.ome_metadata = self._update_pixel_sizes(
                self.ome_metadata, new_x, new_y, new_z
            )
        return True

    @staticmethod
    def _align_mask_depth(masks: np.ndarray, target_z: int) -> np.ndarray:
        """Align mask depth to image depth via nearest-neighbor sampling on Z."""
        if masks.ndim != 3:
            raise ValueError("masks must be 3-D")
        src_z = masks.shape[0]
        if src_z == target_z:
            return masks
        if src_z <= 0 or target_z <= 0:
            raise ValueError(f"Invalid depth src={src_z}, target={target_z}")
        z_idx = np.rint(np.linspace(0, src_z - 1, target_z)).astype(np.int64)
        return masks[z_idx]

    def save_masks(self, path: str | None = None, *, metadata: dict | None = None) -> None:
        """Save current mask stack as a TIFF file."""
        if self.masks is None:
            raise RuntimeError("No masks to save")
        if path is None:
            if self.mask_path is None:
                raise RuntimeError("No path specified for saving masks")
            path = self.mask_path
        desc = json.dumps(metadata, ensure_ascii=False) if metadata else None
        tifffile.imwrite(path, self.masks.astype(np.uint8), description=desc)
        self.mask_path = path
        self.mask_dirty = False

    def replace_masks(
        self,
        masks: np.ndarray | None,
        *,
        components: np.ndarray | None = None,
        mask_path: str | None = None,
        dirty: bool = True,
    ) -> None:
        """Replace the entire mask stack and optionally reuse precomputed component labels."""
        if masks is None:
            self.masks = None
            self.components = None
            self.mask_path = mask_path
            self.mask_dirty = dirty
            self._touch_mask_revision()
            return
        arr = (np.asarray(masks) > 0).astype(np.uint8, copy=False)
        if arr.ndim != 3:
            raise ValueError(f"Mask stack must be 3-D, got shape={arr.shape}")
        if self.data is not None and arr.shape != self.data.shape:
            raise ValueError(f"Mask stack shape {arr.shape} does not match image shape {self.data.shape}")
        self.masks = arr
        self.mask_path = mask_path if mask_path is not None else self.mask_path
        self.mask_dirty = dirty
        if components is not None:
            comp_arr = np.asarray(components)
            if comp_arr.shape != arr.shape:
                raise ValueError(
                    f"Component label shape {comp_arr.shape} does not match mask shape {arr.shape}"
                )
            self.components = comp_arr.astype(np.int32, copy=False)
        else:
            self.update_components()
        self._touch_mask_revision()

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
            self.update_components()
        self._touch_mask_revision()

    # --------- mask helpers ---------
    def default_mask_path(self) -> str:
        if self.path is None:
            raise RuntimeError("Image not loaded")
        base = os.path.splitext(os.path.basename(self.path))[0] + "_mask.tif"
        return os.path.join(os.path.dirname(self.path), base)

    def create_blank_masks(self, path: str | None = None, *, metadata: dict | None = None) -> None:
        if self.data is None:
            raise RuntimeError("Image must be loaded before creating masks")
        if path is None:
            path = self.default_mask_path()
        self.masks = np.zeros_like(self.data, dtype=np.uint8)
        self.mask_path = path
        desc = json.dumps(metadata) if metadata else None
        tifffile.imwrite(self.mask_path, self.masks, description=desc)
        self.mask_dirty = False
        self.components = np.zeros_like(self.masks, dtype=np.int32)
        self._touch_mask_revision()

    def save_slice(self, slice_idx: int | None = None) -> None:
        """Write ``slice_idx`` of the mask stack back to ``mask_path``."""
        if self.masks is None:
            raise RuntimeError("No masks to save")
        if self.mask_path is None:
            raise RuntimeError("No path specified for saving masks")
        if slice_idx is None:
            slice_idx = self.index
        mm = tifffile.memmap(self.mask_path, mode="r+")
        mm[slice_idx] = self.masks[slice_idx].astype(np.uint8)
        mm.flush()
        self.mask_dirty = False

    def ensure_masks(self) -> None:
        """Ensure that an in-memory mask stack exists without writing to disk."""
        if self.data is None:
            raise RuntimeError("Image must be loaded before creating masks")
        if self.masks is None:
            self.masks = np.zeros_like(self.data, dtype=np.uint8)
            self.components = np.zeros_like(self.masks, dtype=np.int32)
            self.mask_dirty = False
            self._touch_mask_revision()

    def update_components(self) -> None:
        """Recompute connected component labels for all masks."""
        if self.masks is None:
            self.components = None
            return
        self.components = label_components(self.masks)

    # 便利属性
    @property
    def n_slices(self) -> int:
        return 0 if self.data is None else self.data.shape[0]

    def get_current(self) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("No image loaded")
        if self.show_original and self.original_data is not None:
            return self.original_data[self.index]
        return self.data[self.index]

    def get_original_slice(self, slice_idx: int | None = None) -> np.ndarray:
        """Return slice from ``original_data`` falling back to ``data``."""
        if slice_idx is None:
            slice_idx = self.index
        if self.original_data is not None:
            return self.original_data[slice_idx]
        if self.data is None:
            raise RuntimeError("Image not loaded")
        return self.data[slice_idx]

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
        return int(self.components.max())

    # -------- intensity helpers ---------
    def _compute_slice_intensity(self) -> np.ndarray:
        if self.original_data is None:
            raise RuntimeError("Image not loaded")
        if self._slice_intensity is None:
            flat = self.original_data.reshape(self.n_slices, -1)
            self._slice_intensity = flat.mean(axis=1)
        return self._slice_intensity

    def get_segment_mask(self, percentile: float = 5.0, continuous: bool = True) -> np.ndarray:
        """Return boolean mask of slices considered worth segmenting."""
        params = (percentile, continuous)
        if self._segment_mask is None or self._seg_params != params:
            if self.original_data is None:
                raise RuntimeError("Image not loaded")
            mask = []
            for slice_ in self.original_data:
                low = np.percentile(slice_, percentile)
                high = np.percentile(slice_, 100 - percentile)
                mask.append(high > low)
            mask = np.array(mask, dtype=bool)
            if continuous and mask.any():
                first = mask.argmax()
                last = len(mask) - 1 - mask[::-1].argmax()
                mask[first : last + 1] = True
            self._segment_mask = mask
            self._seg_params = params
        return self._segment_mask

    # --------- slice helpers ---------
    def _extract_slice(self, idx: int) -> np.ndarray:
        """Return slice ``idx`` from ``data``."""
        if self.data is None:
            raise RuntimeError("Image not loaded")
        if not (0 <= idx < self.n_slices):
            raise ValueError("slice index out of range")
        return self.data[idx]

    def _recompute_image(self) -> None:
        """Recompute ``data`` from ``original_data`` using blur and stretch."""
        if self.original_data is None:
            return
        img = self.original_data
        if self.reverse:
            img = img.max() - img
        if self.blur_sigma > 0:
            img = gaussian_blur_stack(img, self.blur_sigma)
        if self.stretch_percent > 0:
            img = histogram_stretch_stack(img, self.stretch_percent)
        self.data = img

    @staticmethod
    def _normalize_to_8bit(arr: np.ndarray) -> np.ndarray:
        """Normalize array to uint8 range."""
        arr = arr.astype(float)
        mn = arr.min()
        mx = arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = np.clip(arr * 255, 0, 255)
        return arr.astype(np.uint8)

    # --------- image utilities ---------
    def histogram_stretch(self, percentile: float) -> None:
        """Apply histogram stretch to the processed image stack."""
        if self.original_data is None:
            raise RuntimeError("Image not loaded")
        pct = float(percentile)
        if not np.isfinite(pct):
            pct = 0.0
        # Avoid degenerate ranges near/above 50%.
        pct = min(max(pct, 0.0), 49.9)
        self.stretch_percent = pct
        self._recompute_image()

    def reset_contrast(self) -> None:
        """Revert ``data`` to the original loaded image."""
        if self.original_data is not None:
            self.stretch_percent = 0.0
            self._recompute_image()

    def apply_gaussian_blur(self, sigma: float) -> None:
        """Apply Gaussian blur to the image stack for processing."""
        if self.original_data is None:
            return
        self.blur_sigma = sigma
        self._recompute_image()

    def remove_gaussian_blur(self) -> None:
        """Restore the image stack to the state before blurring."""
        if self.blur_sigma == 0.0:
            return
        self.blur_sigma = 0.0
        self._recompute_image()

    def toggle_show_original(self) -> None:
        """Toggle display between processed and original image."""
        if self.original_data is None:
            return
        self.show_original = not self.show_original

    def toggle_reverse_intensity(self) -> None:
        """Toggle display of pixel intensities reversed relative to max."""
        if self.original_data is None:
            return
        self.reverse = not self.reverse
        self._recompute_image()

    def remove_background(
        self,
        percentile: float,
        bins: int = 0,
        slice_idx: int | None = None,
        *,
        progress: bool = False,
        progress_fn: Callable | None = None,
    ) -> None:
        """Remove low intensity pixels from the mask on ``slice_idx``."""
        if (self.data is None and self.original_data is None) or self.masks is None:
            return
        if slice_idx is None:
            slice_idx = self.index
        img = self._extract_slice(slice_idx)
        mask = self.masks[slice_idx]

        global_thresh = None
        if bins and bins > 0:
            values = self.data[self.masks > 0] if self.data is not None else np.array([])
            if values.size > 0:
                hist, edges = np.histogram(values, bins=256)
                idx = min(bins, len(edges) - 2)
                global_thresh = float(edges[idx])

        new_mask = remove_mask_background(
            img,
            mask,
            percentile,
            global_thresh,
            progress=progress,
            progress_fn=progress_fn,
        )
        self.set_mask(new_mask, slice_idx)

    def threshold_absolute(self, value: float, slice_idx: int | None = None) -> None:
        """Mark pixels above ``value`` without altering existing foreground."""
        if self.data is None and self.original_data is None:
            return
        if slice_idx is None:
            slice_idx = self.index
        slice_ = self._extract_slice(slice_idx)
        thresh_mask = threshold_absolute(slice_, value)
        if self.masks is None:
            mask = thresh_mask
        else:
            mask = self.get_mask(slice_idx).copy()
            mask[(mask == 0) & (thresh_mask > 0)] = 1
        self.set_mask(mask, slice_idx)

    def threshold_normalized(self, percent: float, slice_idx: int | None = None) -> None:
        """Threshold slice by normalized percentage without removing labels."""
        if self.data is None and self.original_data is None:
            return
        if slice_idx is None:
            slice_idx = self.index
        slice_ = self._extract_slice(slice_idx)
        thresh_mask = threshold_normalized(slice_, percent)
        if self.masks is None:
            mask = thresh_mask
        else:
            mask = self.get_mask(slice_idx).copy()
            mask[(mask == 0) & (thresh_mask > 0)] = 1
        self.set_mask(mask, slice_idx)

    # --------- utility methods ---------
    def delete_components_touching_rect(
        self, slice_idx: int, x0: int, y0: int, x1: int, y1: int
    ) -> bool:
        """Delete components that have any pixel within ``(x0,y0,x1,y1)``."""
        if self.masks is None:
            return False
        if self.components is None:
            self.update_components()
        mask = self.masks[slice_idx]
        labels = self.components[slice_idx]
        height, width = mask.shape
        x0 = max(0, min(int(x0), width))
        x1 = max(0, min(int(x1), width))
        y0 = max(0, min(int(y0), height))
        y1 = max(0, min(int(y1), height))
        if x0 >= x1 or y0 >= y1:
            return False
        sub = labels[y0:y1, x0:x1]
        to_del = np.unique(sub)
        to_del = to_del[to_del > 0]
        if to_del.size == 0:
            return False
        new_mask = mask.copy()
        for lbl in to_del:
            new_mask[labels == lbl] = 0
        self.set_mask(new_mask, slice_idx)
        return True

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
        self._touch_image_revision()
        if self.masks is not None:
            self._touch_mask_revision()

    def save_stack(self, path: str) -> None:
        """Save current image stack with OME metadata if available."""
        if self.data is None:
            raise RuntimeError("No image loaded")
        tifffile.imwrite(path, self.data, ome=self.ome_metadata)

    # ----- resolution helpers -----
    def get_pixel_sizes(self) -> tuple[float, float, float] | None:
        """Return physical pixel sizes (X, Y, Z) from OME metadata."""
        if not self.ome_metadata:
            return None
        try:
            root = ET.fromstring(self.ome_metadata)
            pixels = self._find_pixels_element(root)
            if pixels is None:
                return None
            x = float(pixels.attrib.get('PhysicalSizeX', '1'))
            y = float(pixels.attrib.get('PhysicalSizeY', '1'))
            z = float(pixels.attrib.get('PhysicalSizeZ', '1'))
            return (x, y, z)
        except Exception:
            return None

    @staticmethod
    def _find_pixels_element(root: ET.Element) -> ET.Element | None:
        """Find OME Pixels element regardless of XML namespace."""
        for elem in root.iter():
            tag = elem.tag
            if isinstance(tag, str) and tag.rsplit('}', 1)[-1] == 'Pixels':
                return elem
        return None

    @staticmethod
    def _update_pixel_sizes(
        ome_xml: str | None, x: float, y: float, z: float
    ) -> str | None:
        if ome_xml is None:
            return None
        try:
            root = ET.fromstring(ome_xml)
            pixels = ZStackModel._find_pixels_element(root)
            if pixels is not None:
                pixels.set('PhysicalSizeX', str(x))
                pixels.set('PhysicalSizeY', str(y))
                pixels.set('PhysicalSizeZ', str(z))
            return ET.tostring(root, encoding='unicode')
        except Exception:
            return ome_xml

    def save_resampled_stack(
        self, path: str, x: float, y: float, z: float
    ) -> None:
        """Save a resampled copy of the stack with new pixel sizes."""
        if self.data is None:
            raise RuntimeError("No image loaded")
        current = self.get_pixel_sizes()
        if current is None:
            raise RuntimeError("Pixel size info missing in metadata")
        sx = current[0] / x
        sy = current[1] / y
        sz = current[2] / z
        new_stack = zoom(self.data, (sz, sy, sx), order=1)
        ome_xml = self._update_pixel_sizes(self.ome_metadata, x, y, z)
        tifffile.imwrite(path, new_stack, ome=ome_xml)
