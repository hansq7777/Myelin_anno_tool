from __future__ import annotations

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel

import numpy as np
from ..utils.dialogs import question_with_shortcuts
from ..utils import config
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .main_controller import MainController


class FileOpsMixin:
    """Helper methods for loading and saving stacks and masks."""

    def _open_file(self: 'MainController') -> None:
        if not self._prompt_save_if_dirty():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF", "", "TIFF Images (*.tif *.tiff *.ome.tif)"
        )
        if path:
            self.model.load(path)
            self.slider.setRange(0, self.model.n_slices - 1)
            self.slider.setEnabled(True)
            mask_folder = config.get("mask_folder")
            if mask_folder:
                base = os.path.splitext(os.path.basename(path))[0] + '_mask.tif'
                mask_path = os.path.join(mask_folder, base)
                if os.path.exists(mask_path):
                    try:
                        self.model.load_masks(mask_path)
                    except Exception:
                        pass
            self._update_view(reset_view=True)

    def _open_masks(self: 'MainController') -> None:
        if not self._prompt_save_if_dirty():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Mask Stack", "", "TIFF Images (*.tif *.tiff)"
        )
        if path:
            self.model.load_masks(path)
            self._update_view()

    def _save_masks(self: 'MainController') -> None:
        if self.model.masks is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Masks",
            self.model.mask_path
            or (self.model.default_mask_path() if self.model.data is not None else ""),
            "TIFF Images (*.tif)",
        )
        if path:
            self.model.save_masks(path)
            self._update_file_labels()

    def _quick_save_masks(self: 'MainController') -> None:
        if self.model.masks is None:
            return
        if self.model.mask_path is None:
            self._save_masks()
        else:
            self.model.save_masks()
        self._update_file_labels()

    def _create_masks(self: 'MainController') -> None:
        if self.model.data is None:
            return
        default_folder = os.path.dirname(self.model.path) if self.model.path else ""
        folder = QFileDialog.getExistingDirectory(self, "Create Mask Stack", default_folder)
        if folder:
            base = os.path.splitext(os.path.basename(self.model.path))[0] + "_mask.tif"
            path = os.path.join(folder, base)
            self.model.create_blank_masks(path)
            self._update_view()

    def _prompt_save_if_dirty(self: 'MainController') -> bool:
        if not self.model.mask_dirty:
            return True
        ret = question_with_shortcuts(self, "Save Masks", "Save mask changes?")
        if ret == QMessageBox.Cancel:
            return False
        if ret == QMessageBox.Yes:
            if self.model.mask_path is None:
                self._save_masks()
            else:
                self.model.save_masks()
        return True

    def choose_annotation_folder(self: 'MainController') -> str | None:
        path = QFileDialog.getExistingDirectory(self, "Select Annotation Folder")
        return path or None

    def _set_mask_folder(self: 'MainController') -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Mask Folder", config.get("mask_folder", "")
        )
        if folder:
            config.set("mask_folder", folder)

    def close_current(self: 'MainController') -> None:
        self.model = self.model.__class__()
        self.slider.setEnabled(False)
        self.canvas.set_image(np.zeros((1, 1), dtype=np.uint8), reset_view=True)
        self.canvas.set_mask(None)
        self.statusBar().showMessage("Ready")

    def _short_path(self: 'MainController', path: str | None) -> str:
        if not path:
            return "(none)"
        parent = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)
        return os.path.join(parent, name)

    def _update_file_labels(self: 'MainController') -> None:
        if hasattr(self, "image_label"):
            self.image_label.setText(f"Image: {self._short_path(self.model.path)}")
        if hasattr(self, "mask_label"):
            self.mask_label.setText(f"Mask: {self._short_path(self.model.mask_path)}")

