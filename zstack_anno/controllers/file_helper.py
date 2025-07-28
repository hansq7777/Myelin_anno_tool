from __future__ import annotations

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel

import numpy as np
from ..utils.dialogs import question_with_shortcuts
from ..utils import config
from ..utils.czi_utils import (
    split_czi_file,
    czi_to_tiff,
    CziNotSupportedError,
)
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

    # --------- additional file helpers ---------
    def _import_czi_file(self: 'MainController') -> None:
        """Import a CZI file and convert it to OME-TIFF."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import CZI", "", "CZI Files (*.czi)"
        )
        if not path:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", os.path.dirname(path)
        )
        if not out_dir:
            return

        ret = QMessageBox.question(
            self,
            "Import CZI",
            (
                "Split stacks based on stage coordinates?\n"
                "Choose No to keep all data in a single stack."
            ),
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if ret == QMessageBox.Cancel:
            return

        try:
            if ret == QMessageBox.Yes:
                written = split_czi_file(path, out_dir)
            else:
                name = os.path.splitext(os.path.basename(path))[0] + ".ome.tif"
                out_path = os.path.join(out_dir, name)
                written = [czi_to_tiff(path, out_path)]
        except CziNotSupportedError as exc:
            QMessageBox.warning(self, "CZI Support Missing", str(exc))
            return

        QMessageBox.information(
            self,
            "Import Complete",
            f"Written {len(written)} stack(s) to {out_dir}",
        )

    def _export_czi_metadata(self: 'MainController') -> None:
        """Save raw CZI metadata to a file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CZI", "", "CZI Files (*.czi)"
        )
        if not path:
            return
        default = os.path.splitext(os.path.basename(path))[0] + "_meta.xml"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Metadata",
            os.path.join(os.path.dirname(path), default),
            "XML Files (*.xml)"
        )
        if not out_path:
            return
        try:
            from ..utils.czi_utils import dump_czi_metadata

            dump_czi_metadata(path, out_path)
        except CziNotSupportedError as exc:
            QMessageBox.warning(self, "CZI Support Missing", str(exc))
            return

        QMessageBox.information(
            self,
            "Metadata Saved",
            f"Metadata written to {out_path}"
        )

    def _show_stack_info(self: 'MainController') -> None:
        """Display information about the currently loaded stack."""
        if self.model.data is None:
            return
        shape = self.model.data.shape
        pixels = self.model.get_pixel_sizes()
        if pixels is None:
            px_info = "unknown"
        else:
            px_info = f"{pixels[0]} x {pixels[1]} x {pixels[2]} µm"
        msg = (
            f"Dimensions: {shape}\n"
            f"Pixel size: {px_info}"
        )
        QMessageBox.information(self, "Stack Info", msg)

