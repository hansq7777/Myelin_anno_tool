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
)
from PyQt5.QtGui import QTransform
from PyQt5.QtCore import Qt

from ..pipeline import run_strategy, overlay_image, read_stack
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
        self._metrics: List[tuple[float, float]] = []
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
        self._metrics.clear()
        self._names.clear()
        n = self._stack.shape[0]
        # block signals while adjusting to avoid premature _update_images call
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, n - 1)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)
        self.slice_slider.setEnabled(True)
        self._build_panels()
        self._update_images()

    # ---- Processing ----
    def _run(self) -> None:
        self.stack_path = self.stack_edit.text().strip()
        self.gt_path = self.gt_edit.text().strip()
        if not os.path.isfile(self.stack_path) or not os.path.isfile(self.gt_path):
            return
        self._stack = read_stack(self.stack_path)
        self._gt = read_stack(self.gt_path).astype(np.uint8)
        self._preds.clear()
        self._metrics.clear()
        self._names.clear()
        slice_idx = self.slice_slider.value() if self.slice_radio.isChecked() else None
        for i in range(self.strat_list.count()):
            item = self.strat_list.item(i)
            path = item.data(Qt.UserRole)
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                with open(path, "r", encoding="utf-8") as f:
                    steps = json.load(f)
            except Exception:
                continue
            pred, prec, rec = run_strategy(
                self.stack_path,
                self.gt_path,
                steps,
                slice_idx=slice_idx,
            )
            self._preds.append(pred)
            self._metrics.append((prec, rec))
            self._names.append(name)
        if self._stack is not None:
            n = self._stack.shape[0]
            # block signals while adjusting slider to prevent premature updates
            self.slice_slider.blockSignals(True)
            self.slice_slider.setRange(0, n - 1)
            self.slice_slider.setValue(0)
            self.slice_slider.blockSignals(False)
            self.slice_slider.setEnabled(True)
        self._build_panels()
        self._update_images()

    def _build_panels(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        count = max(self.window_spin.value(), len(self._names) + 1)
        self.canvases: List[SyncCanvas] = []
        titles = ["Ground Truth"] + self._names
        titles += [""] * (count - len(titles))

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

    def _sync_views(self, transform: QTransform) -> None:
        sender = self.sender()
        for c in self.canvases:
            if c is sender:
                continue
            c.apply_transform(transform)

    def _update_images(self) -> None:
        if self._stack is None or self._gt is None:
            return
        idx = self.slice_slider.value()
        img = self._stack[idx]
        gt_slice = self._gt[idx]
        # first panel: GT overlay
        base = overlay_image(img, gt_slice, gt_slice, alpha=0.4)
        self.canvases[0].set_image(base)
        for i in range(1, len(self.canvases)):
            if i - 1 < len(self._preds):
                overlay = overlay_image(img, gt_slice, self._preds[i - 1][idx], alpha=0.4)
                self.canvases[i].set_image(overlay)
                prec, rec = self._metrics[i - 1]
                tooltip = f"precision={prec:.3f} recall={rec:.3f}"
            else:
                self.canvases[i].set_image(np.stack([ZStackModel._normalize_to_8bit(img)] * 3, axis=-1))
                tooltip = ""
            self.canvases[i].setToolTip(tooltip)

