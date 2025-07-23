import json
import os
from typing import List, Optional

import numpy as np
import tifffile
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QSpinBox,
    QScrollArea,
    QWidget,
    QGridLayout,
    QMenu,
    QMessageBox,
)
from PyQt5.QtGui import QTransform
from PyQt5.QtCore import Qt

from ..pipeline import overlay_mask, overlay_image, compute_metrics, read_stack
from ..views.canvas import SyncCanvas
from ..models.zstack_model import ZStackModel


class MetaCanvas(SyncCanvas):
    """Canvas that can show metadata on right-click."""

    def __init__(self) -> None:
        super().__init__()
        self.metadata: Optional[dict] = None

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        menu = QMenu(self)
        info_act = menu.addAction("Show Metadata")
        chosen = menu.exec_(self.mapToGlobal(event.pos()))
        if chosen is info_act and self.metadata:
            steps = self.metadata.get("steps")
            strat = self.metadata.get("strategy", "")
            if isinstance(steps, list):
                lines = [f"{i+1}. {s.get('action', '')}" for i, s in enumerate(steps)]
                steps_text = "\n".join(lines)
            else:
                steps_text = str(steps)
            text = f"Strategy: {strat}\nSteps:\n{steps_text}" if strat else steps_text
            QMessageBox.information(self, "Metadata", text)


class ValidationDialog(QDialog):
    """View existing segmentation results alongside ground truth."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Validation Viewer")
        self.stack_path = ""
        self.gt_path = ""
        self._stack: Optional[np.ndarray] = None
        self._gt: Optional[np.ndarray] = None
        self._masks: List[np.ndarray] = []
        self._names: List[str] = []
        self._metas: List[Optional[dict]] = []
        self._metrics: List[Optional[tuple[float, float]]] = []

        layout = QVBoxLayout(self)

        file_layout = QHBoxLayout()
        self.stack_edit = QLineEdit()
        stack_btn = QPushButton("Stack…")
        stack_btn.clicked.connect(self._choose_stack)
        self.gt_edit = QLineEdit()
        gt_btn = QPushButton("Ground Truth…")
        gt_btn.clicked.connect(self._choose_gt)
        file_layout.addWidget(QLabel("Stack:"))
        file_layout.addWidget(self.stack_edit)
        file_layout.addWidget(stack_btn)
        file_layout.addWidget(QLabel("GT:"))
        file_layout.addWidget(self.gt_edit)
        file_layout.addWidget(gt_btn)
        layout.addLayout(file_layout)

        mask_layout = QHBoxLayout()
        self.mask_list = QListWidget()
        add_file_btn = QPushButton("Add Files…")
        add_file_btn.clicked.connect(self._add_files)
        add_folder_btn = QPushButton("Add Folder…")
        add_folder_btn.clicked.connect(self._add_folder)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._remove_mask)
        mask_layout.addWidget(self.mask_list)
        btns = QVBoxLayout()
        btns.addWidget(add_file_btn)
        btns.addWidget(add_folder_btn)
        btns.addWidget(rm_btn)
        btns.addStretch()
        mask_layout.addLayout(btns)
        layout.addLayout(mask_layout)

        ctrl_layout = QHBoxLayout()
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self._update_images)
        ctrl_layout.addWidget(self.slice_slider)
        ctrl_layout.addWidget(QLabel("Windows:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 9)
        self.window_spin.setValue(4)
        self.window_spin.valueChanged.connect(self._build_panels)
        ctrl_layout.addWidget(self.window_spin)
        self.stats_btn = QPushButton("Whole Stack Stats")
        self.stats_btn.clicked.connect(self._show_stats)
        ctrl_layout.addWidget(self.stats_btn)
        layout.addLayout(ctrl_layout)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.scroll.setWidget(self.grid_widget)
        layout.addWidget(self.scroll)

        self.legend_widget = QWidget()
        legend_layout_inner = QHBoxLayout(self.legend_widget)
        for text, color in [
            ("TP", (0, 158, 115)),
            ("FP", (213, 94, 0)),
            ("FN", (0, 114, 178)),
        ]:
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background-color: rgb({color[0]}, {color[1]}, {color[2]});"
            )
            legend_layout_inner.addWidget(swatch)
            legend_layout_inner.addWidget(QLabel(text))
        legend_layout_inner.addStretch()
        layout.addWidget(self.legend_widget)
        self.legend_widget.hide()

    # ---- UI helpers ----
    def _choose_stack(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Stack", "", "TIFF (*.tif *.tiff *.ome.tif)")
        if path:
            self.stack_edit.setText(path)
            self._load_stack()

    def _choose_gt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth", "", "TIFF (*.tif *.tiff *.ome.tif)")
        if path:
            self.gt_edit.setText(path)
            self._load_gt()

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Mask Files", "", "TIFF (*.tif *.tiff *.ome.tif)")
        for path in paths:
            self._add_mask_path(path)
        if paths:
            self._load_masks()

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        base = os.path.splitext(os.path.basename(self.stack_edit.text().strip()))[0]
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".tif", ".tiff", ".ome.tif")):
                continue
            if base and not fname.startswith(base):
                continue
            self._add_mask_path(os.path.join(folder, fname))
        self._load_masks()

    def _add_mask_path(self, path: str) -> None:
        for i in range(self.mask_list.count()):
            if self.mask_list.item(i).data(Qt.UserRole) == path:
                return
        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.UserRole, path)
        self.mask_list.addItem(item)

    def _remove_mask(self) -> None:
        row = self.mask_list.currentRow()
        if row >= 0:
            self.mask_list.takeItem(row)
            self._load_masks()

    def _load_stack(self) -> None:
        path = self.stack_edit.text().strip()
        if not os.path.isfile(path):
            return
        self.stack_path = path
        self._stack = read_stack(path)
        n = self._stack.shape[0]
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, n - 1)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)
        self.slice_slider.setEnabled(True)
        self._update_images()

    def _load_gt(self) -> None:
        path = self.gt_edit.text().strip()
        if not os.path.isfile(path):
            return
        self.gt_path = path
        self._gt = read_stack(path).astype(np.uint8)
        self._load_masks()

    def _load_masks(self) -> None:
        self._masks.clear()
        self._names.clear()
        self._metas.clear()
        self._metrics.clear()
        if self._gt is not None:
            self._masks.append(self._gt)
            self._names.append("Ground Truth")
            self._metas.append({"strategy": "Ground Truth"})
            self._metrics.append((1.0, 1.0))
        for i in range(self.mask_list.count()):
            path = self.mask_list.item(i).data(Qt.UserRole)
            if not path or not os.path.isfile(path):
                continue
            arr = read_stack(path).astype(np.uint8)
            meta = None
            try:
                with tifffile.TiffFile(path) as tif:
                    desc = tif.pages[0].description
                    if desc:
                        meta = json.loads(desc)
            except Exception:
                meta = None
            name = ""
            if meta and isinstance(meta, dict):
                name = meta.get("strategy", "")
            if not name:
                name = os.path.splitext(os.path.basename(path))[0]
            self._masks.append(arr)
            self._names.append(name)
            self._metas.append(meta)
            if self._gt is not None and arr.shape == self._gt.shape:
                self._metrics.append(compute_metrics(self._gt, arr))
            else:
                self._metrics.append(None)
        self._build_panels()
        self.legend_widget.setVisible(self._gt is not None and len(self._masks) > 1)

    # ---- Display helpers ----
    def _build_panels(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        count = max(self.window_spin.value(), len(self._masks))
        self.canvases: List[MetaCanvas] = []
        for i in range(count):
            canvas = MetaCanvas()
            label_text = self._names[i] if i < len(self._names) else ""
            if i > 0 and i < len(self._metrics) and self._metrics[i] is not None:
                p, r = self._metrics[i]
                label_text += f"\nP:{p:.2f} R:{r:.2f}"
            lbl = QLabel(label_text)
            lbl.setAlignment(Qt.AlignHCenter)
            widget = QWidget()
            wrapper = QVBoxLayout(widget)
            wrapper.addWidget(canvas)
            wrapper.addWidget(lbl)
            row, col = divmod(i, 3)
            self.grid.addWidget(widget, row, col)
            canvas.viewChanged.connect(self._sync_views)
            if i < len(self._metas):
                canvas.metadata = self._metas[i]
            self.canvases.append(canvas)
        self._update_images()

    def _sync_views(self, transform: QTransform, h: int, v: int) -> None:
        sender = self.sender()
        for c in self.canvases:
            if c is sender:
                continue
            c.apply_transform(transform, h, v)

    def _update_images(self) -> None:
        if self._stack is None or not self.canvases:
            return
        idx = self.slice_slider.value()
        img = self._stack[idx]
        for i, canvas in enumerate(self.canvases):
            if i < len(self._masks):
                if i == 0 or self._gt is None:
                    overlay = overlay_mask(img, self._masks[i][idx], alpha=0.4)
                else:
                    overlay = overlay_image(img, self._gt[idx], self._masks[i][idx], alpha=0.4)
                canvas.set_image(overlay)
            else:
                canvas.set_image(
                    np.stack([ZStackModel._normalize_to_8bit(img)] * 3, axis=-1)
                )
            canvas.setToolTip("")

    def _show_stats(self) -> None:
        if self._gt is None or len(self._masks) <= 1:
            QMessageBox.information(
                self,
                "Statistics",
                "Load ground truth and at least one mask to compute stats.",
            )
            return
        lines = []
        for name, mask in zip(self._names[1:], self._masks[1:]):
            if mask.shape != self._gt.shape:
                lines.append(f"{name}: shape mismatch")
                continue
            p, r = compute_metrics(self._gt, mask)
            lines.append(f"{name}: precision={p:.3f} recall={r:.3f}")
        QMessageBox.information(self, "Statistics", "\n".join(lines))

