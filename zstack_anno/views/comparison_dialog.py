import json
import os
from typing import List

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
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QScrollArea,
    QWidget,
    QGridLayout,
    QInputDialog,
)
from PyQt5.QtGui import QTransform
from PyQt5.QtCore import Qt

from ..pipeline import run_strategy, overlay_mask, read_stack
from ..views.canvas import SyncCanvas
from ..models.zstack_model import ZStackModel


class ComparisonDialog(QDialog):
    """Visualize segmentation strategies against ground truth."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Strategy Comparison")
        self.stack_path = ""
        self.gt_path = ""
        self._preds: List[np.ndarray] = []
        self._names: List[str] = []
        self._stack: np.ndarray | None = None
        self._gt: np.ndarray | None = None

        layout = QVBoxLayout(self)

        # ---- File selection ----
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

        # ---- Strategy list ----
        strat_layout = QHBoxLayout()
        self.strat_list = QListWidget()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_strategy)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._remove_strategy)
        strat_layout.addWidget(self.strat_list)
        btns = QVBoxLayout()
        btns.addWidget(add_btn)
        btns.addWidget(rm_btn)
        btns.addStretch()
        strat_layout.addLayout(btns)
        layout.addLayout(strat_layout)

        # ---- Controls ----
        ctrl_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self._update_images)
        self.whole_radio = QRadioButton("Whole Z Stack")
        self.slice_radio = QRadioButton("Single Slice")
        self.whole_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.whole_radio)
        group.addButton(self.slice_radio)
        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.whole_radio)
        ctrl_layout.addWidget(self.slice_radio)
        ctrl_layout.addWidget(self.slice_slider)
        ctrl_layout.addWidget(QLabel("Windows:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 9)
        self.window_spin.setValue(4)
        self.window_spin.valueChanged.connect(self._build_panels)
        ctrl_layout.addWidget(self.window_spin)
        layout.addLayout(ctrl_layout)

        # ---- Results area ----
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.scroll.setWidget(self.grid_widget)
        layout.addWidget(self.scroll)

    # ---- UI helpers ----
    def _choose_stack(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Stack", "", "TIFF (*.tif *.tiff *.ome.tif)")
        if path:
            self.stack_edit.setText(path)
            self._load_preview()

    def _choose_gt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth", "", "TIFF (*.tif *.tiff *.ome.tif)")
        if path:
            self.gt_edit.setText(path)
            self._load_preview()

    def _add_strategy(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Strategy", "", "JSON (*.json)")
        if path:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            self.strat_list.addItem(item)

    def _remove_strategy(self) -> None:
        row = self.strat_list.currentRow()
        if row >= 0:
            self.strat_list.takeItem(row)

    def _load_preview(self) -> None:
        """Load stack and ground truth for preview if both paths are set."""
        self.stack_path = self.stack_edit.text().strip()
        self.gt_path = self.gt_edit.text().strip()
        if not os.path.isfile(self.stack_path) or not os.path.isfile(self.gt_path):
            return
        self._stack = read_stack(self.stack_path)
        self._gt = read_stack(self.gt_path).astype(np.uint8)
        self._preds.clear()
        self._names.clear()
        n = self._stack.shape[0]
        # block signals while adjusting to avoid premature _update_images call
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, n - 1)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)
        self.slice_slider.setEnabled(True)
        self._build_panels()

    # ---- Processing ----
    def _run(self) -> None:
        self.stack_path = self.stack_edit.text().strip()
        self.gt_path = self.gt_edit.text().strip()
        if not os.path.isfile(self.stack_path) or not os.path.isfile(self.gt_path):
            return
        self._stack = read_stack(self.stack_path)
        self._gt = read_stack(self.gt_path).astype(np.uint8)
        self._preds.clear()
        self._names.clear()

        paths: list[str] = []
        for i in range(self.strat_list.count()):
            item = self.strat_list.item(i)
            path = item.data(Qt.UserRole)
            if path:
                paths.append(path)
                name = os.path.splitext(os.path.basename(path))[0]
                self._names.append(name)

        slice_idx = self.slice_slider.value() if self.slice_radio.isChecked() else None
        self._build_panels()

        for si, path in enumerate(paths):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    steps = json.load(f)
            except Exception:
                continue

            pred, _, _ = run_strategy(
                self.stack_path,
                self.gt_path,
                steps,
                slice_idx=slice_idx,
                step_callback=lambda m, sidx, step_idx, s=si: self._live_update(s, m, sidx),
            )

            self._preds.append(pred)

            base = os.path.splitext(os.path.basename(self.stack_path))[0]
            if slice_idx is None:
                default = os.path.join(os.path.dirname(self.stack_path), f"{base}_{self._names[si]}.tif")
            else:
                default = os.path.join(
                    os.path.dirname(self.stack_path), f"{base}_slice{slice_idx + 1}_{self._names[si]}.tif"
                )
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Result", default, "TIFF Images (*.tif)")
            if save_path:
                if slice_idx is None:
                    tifffile.imwrite(save_path, pred)
                else:
                    tifffile.imwrite(save_path, pred[slice_idx])

        if self._stack is not None:
            n = self._stack.shape[0]
            self.slice_slider.blockSignals(True)
            self.slice_slider.setRange(0, n - 1)
            self.slice_slider.setValue(0)
            self.slice_slider.blockSignals(False)
            self.slice_slider.setEnabled(True)

        self._update_images()

    def _build_panels(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        count = self.window_spin.value()

        # if the requested window count is less than the number of images to
        # display, ask the user which strategies to hide until it fits
        while len(self._names) > count:
            choice, ok = QInputDialog.getItem(
                self,
                "Close Window",
                "Select a strategy to hide:",
                self._names,
                0,
                False,
            )
            if not ok:
                # user cancelled -> keep all windows
                count = len(self._names)
                self.window_spin.setValue(count)
                break
            idx = self._names.index(choice)
            del self._names[idx]
            del self._preds[idx]

        count = max(count, len(self._names))
        self.canvases: List[SyncCanvas] = []
        titles = self._names + [""] * (count - len(self._names))

        for i in range(count):
            canvas = SyncCanvas()
            lbl = QLabel(titles[i])
            lbl.setAlignment(Qt.AlignHCenter)
            wrapper = QVBoxLayout()
            widget = QWidget()
            wrapper.addWidget(canvas)
            wrapper.addWidget(lbl)
            widget.setLayout(wrapper)
            row, col = divmod(i, 3)
            self.grid.addWidget(widget, row, col)
            self.canvases.append(canvas)

        for c in self.canvases:
            c.viewChanged.connect(self._sync_views)

        # after rebuilding panels update the images shown
        self._update_images()

    def _sync_views(self, transform: QTransform, h: int, v: int) -> None:
        sender = self.sender()
        for c in self.canvases:
            if c is sender:
                continue
            c.apply_transform(transform, h, v)

    def _update_images(self) -> None:
        if self._stack is None:
            return
        idx = self.slice_slider.value()
        img = self._stack[idx]
        for i, canvas in enumerate(self.canvases):
            if i < len(self._preds):
                overlay = overlay_mask(img, self._preds[i][idx], alpha=0.4)
                canvas.set_image(overlay)
            else:
                canvas.set_image(
                    np.stack([ZStackModel._normalize_to_8bit(img)] * 3, axis=-1)
                )
            canvas.setToolTip("")

    def _live_update(self, strat_idx: int, masks: np.ndarray, slice_idx: int) -> None:
        """Update a canvas during strategy execution."""
        if self._stack is None or strat_idx >= len(self.canvases):
            return
        if strat_idx < len(self._preds):
            self._preds[strat_idx] = masks
        else:
            # when called before final append
            self._preds.append(masks)
        if slice_idx == self.slice_slider.value():
            img = self._stack[slice_idx]
            overlay = overlay_mask(img, masks[slice_idx], alpha=0.4)
            self.canvases[strat_idx].set_image(overlay)

