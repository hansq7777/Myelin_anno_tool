from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QApplication
from PyQt5.QtCore import QThread, pyqtSignal
import threading
import os

from ..utils import morphology_tools
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .main_controller import MainController


class IntGrowThread(QThread):
    finished = pyqtSignal(np.ndarray)
    cancelled = pyqtSignal()
    progress = pyqtSignal(np.ndarray, int, int)

    def __init__(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        diff: float,
        hist: float | None,
        force: float | None,
        event: threading.Event,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        self.img = img
        self.mask = mask
        self.diff = diff
        self.hist = hist
        self.force = force
        self.event = event
        self.limit = limit
        self._next = 0.2

    def _progress_cb(self, cur: int, total: int, mask: np.ndarray | None = None) -> None:
        if mask is None or total == 0:
            return
        frac = cur / float(total)
        if frac >= self._next:
            self.progress.emit(mask.copy(), cur, total)
            self._next += 0.2

    def run(self) -> None:
        result = morphology_tools.intensity_region_grow(
            self.img,
            self.mask,
            self.diff,
            self.hist,
            self.force,
            self.limit,
            progress=True,
            progress_fn=self._progress_cb,
            cancel_event=self.event,
        )
        if self.event.is_set():
            self.cancelled.emit()
        else:
            self.finished.emit(result)


class MorphologyMixin:
    """Actions for modifying masks and images."""

    def _dilate_current(self: 'MainController') -> None:
        if not self._ensure_masks():
            return
        self._push_undo("dilate")
        cur = self.model.get_mask()
        iterations = self.strength_spin.value() if hasattr(self, "strength_spin") else 1
        new = morphology_tools.dilate(cur, iterations=iterations)
        self.model.set_mask(new)
        self._update_view()

    def _erode_current(self: 'MainController') -> None:
        if not self._ensure_masks():
            return
        self._push_undo("erode")
        cur = self.model.get_mask()
        iterations = self.strength_spin.value() if hasattr(self, "strength_spin") else 1
        new = morphology_tools.erode(cur, iterations=iterations)
        self.model.set_mask(new)
        self._update_view()

    def _filter_small(self: 'MainController') -> None:
        if not self._ensure_masks():
            return
        self._push_undo("filter")
        cur = self.model.get_mask()
        thresh = self.filter_spin.value() if hasattr(self, "filter_spin") else 100
        new = morphology_tools.remove_small(cur, thresh)
        self.model.set_mask(new)
        self._update_view()

    def _skeletonize_current(self: 'MainController') -> None:
        if not self._ensure_masks():
            return
        algorithms = ["skeletonize", "skeletonize_3d", "medial_axis"]
        alg, ok = QInputDialog.getItem(self, "Skeletonize", "Algorithm:", algorithms, 0, False)
        if not ok or not alg:
            return
        params: dict[str, object] = {}
        if alg == "medial_axis":
            choice, ok = QInputDialog.getItem(
                self, "Return Distance", "Return distance?", ["False", "True"], 0, False
            )
            if not ok:
                return
            params["return_distance"] = choice == "True"
        self._push_undo("skeletonize")
        if alg == "skeletonize_3d":
            if self.model.masks is None:
                return
            result = morphology_tools.skeletonize_stack(self.model.masks, algorithm=alg, **params)
            self.model.masks = result
        else:
            cur = self.model.get_mask()
            new = morphology_tools.skeletonize_slice(cur, algorithm=alg, **params)
            self.model.set_mask(new)
        self._update_view()

    def _threshold_abs(self: 'MainController') -> None:
        if not self._ensure_masks():
            return
        try:
            value = float(self.abs_thresh_edit.text())
        except ValueError:
            value = 0.0
        self._push_undo("thresh_abs")
        self.model.threshold_absolute(value)
        self._update_view()

    def _threshold_norm(self: 'MainController') -> None:
        if not self._ensure_masks():
            return
        try:
            pct = float(self.norm_thresh_edit.text())
        except ValueError:
            pct = 50.0
        self._push_undo("thresh_norm")
        self.model.threshold_normalized(pct)
        self._update_view()

    def _apply_bg_filter(self: 'MainController') -> None:
        if not self._ensure_masks():
            return
        try:
            pct = float(self.bg_percentile_edit.text())
        except ValueError:
            pct = 0.0
        try:
            bins = int(self.bg_bins_edit.text())
        except ValueError:
            bins = 0
        self._push_undo("bg_filter")
        self._next_progress = 0.2
        self.model.remove_background(pct, bins, progress=True, progress_fn=self._progress_update)
        self._update_view()

    def _apply_stretch(self: 'MainController') -> None:
        if self.model.data is None:
            return
        try:
            pct = float(self.stretch_edit.text())
        except ValueError:
            pct = 0.0
        self._push_undo("stretch", mask=None)
        if pct <= 0:
            self.model.reset_contrast()
        else:
            self.model.histogram_stretch(pct)
        self._update_view(reset_view=True)

    def _apply_blur(self: 'MainController') -> None:
        if self.model.data is None:
            return
        sigma = float(self.blur_spin.value()) if hasattr(self, "blur_spin") else 1.0
        self.model.apply_gaussian_blur(sigma)
        self._update_view()

    def _toggle_original(self: 'MainController') -> None:
        self.model.toggle_show_original()
        self._update_view()

    def _change_opacity(self: 'MainController') -> None:
        value = self.opacity_slider.value() if hasattr(self, "opacity_slider") else 50
        self.canvas.set_mask_opacity(value / 100.0)
        mask = self.model.get_mask() if self.model.masks is not None else None
        self.canvas.set_mask(mask)

    def _clear_blur(self: 'MainController') -> None:
        self.model.remove_gaussian_blur()
        self._update_view()

    def _reverse_image(self: 'MainController') -> None:
        self.model.toggle_reverse_intensity()
        self._update_view()

    def _resample_stack(self: 'MainController') -> None:
        if self.model.data is None:
            return
        sizes = self.model.get_pixel_sizes()
        if sizes is None:
            QMessageBox.warning(self, "Resample", "Pixel size information not found in metadata")
            return
        x, ok = QInputDialog.getDouble(self, "New X size", "PhysicalSizeX", sizes[0], 0.000001, 1e6, 6)
        if not ok:
            return
        y, ok = QInputDialog.getDouble(self, "New Y size", "PhysicalSizeY", sizes[1], 0.000001, 1e6, 6)
        if not ok:
            return
        z, ok = QInputDialog.getDouble(self, "New Z size", "PhysicalSizeZ", sizes[2], 0.000001, 1e6, 6)
        if not ok:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if not folder:
            return
        base = os.path.splitext(os.path.basename(self.model.path))[0]
        name = f"{base}_x{x}_y{y}_z{z}.tif"
        path = os.path.join(folder, name)
        self.model.save_resampled_stack(path, x, y, z)
        QMessageBox.information(self, "Resample", f"Saved to {path}")

    def _seed_current(self: 'MainController') -> None:
        if not self._ensure_masks():
            return
        try:
            pct = float(self.seed_thresh_edit.text())
        except ValueError:
            pct = 90.0
        self._push_undo("seed")
        img = self.model._extract_slice(self.model.index)
        cur = self.model.get_mask()
        seeds = morphology_tools.sample_seeds(img, pct, num_seeds=20000)
        cur = cur.copy()
        cur[seeds > 0] = 1
        self.model.set_mask(cur)
        self._update_view()

    def _grow_intensity(self: 'MainController') -> None:
        if not self._ensure_masks():
            return
        if getattr(self, "grow_thread", None) is not None and self.grow_thread.isRunning():
            return
        try:
            diff_pct = float(self.int_diff_edit.text())
        except ValueError:
            diff_pct = 20.0
        try:
            hist_pct = float(self.int_hist_edit.text())
        except ValueError:
            hist_pct = None
        self._push_undo("int_grow")
        img = self.model._extract_slice(self.model.index)
        cur = self.model.get_mask()
        self.cancel_event.clear()
        self.int_grow_btn.setEnabled(False)
        self.grow_thread = IntGrowThread(img.astype(float), cur, diff_pct, hist_pct, None, self.cancel_event)
        self.grow_thread.finished.connect(self._int_grow_finished)
        self.grow_thread.cancelled.connect(self._int_grow_cancelled)
        self.grow_thread.progress.connect(self._thread_progress)
        self.grow_thread.start()

    def _int_grow_finished(self: 'MainController', result: np.ndarray) -> None:
        self.model.set_mask(result)
        self._update_view()
        self.int_grow_btn.setEnabled(True)
        self.grow_thread = None

    def _int_grow_cancelled(self: 'MainController') -> None:
        self.int_grow_btn.setEnabled(True)
        self.statusBar().showMessage("Operation cancelled")
        self.grow_thread = None

    def _thread_progress(self: 'MainController', mask: np.ndarray, cur: int, total: int) -> None:
        self._progress_update(cur, total, mask)


    def _clear_foreground(self: 'MainController') -> None:
        """Reset the current mask slice to background."""
        if self.model.masks is None:
            return
        self._push_undo("clear_foreground")
        blank = np.zeros_like(self.model.get_mask())
        self.model.set_mask(blank)
        self._update_view()

