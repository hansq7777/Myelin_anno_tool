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
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from ..pipeline import run_strategy, overlay_image


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
        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.slice_slider)
        layout.addLayout(ctrl_layout)

        # ---- Results area ----
        self.img_layout = QHBoxLayout()
        layout.addLayout(self.img_layout)

    # ---- UI helpers ----
    def _choose_stack(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Stack", "", "TIFF (*.tif *.tiff *.ome.tif)")
        if path:
            self.stack_edit.setText(path)

    def _choose_gt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth", "", "TIFF (*.tif *.tiff *.ome.tif)")
        if path:
            self.gt_edit.setText(path)

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

    # ---- Processing ----
    def _run(self) -> None:
        self.stack_path = self.stack_edit.text().strip()
        self.gt_path = self.gt_edit.text().strip()
        if not os.path.isfile(self.stack_path) or not os.path.isfile(self.gt_path):
            return
        self._stack = tifffile.imread(self.stack_path)
        self._gt = tifffile.imread(self.gt_path).astype(np.uint8)
        self._preds.clear()
        self._metrics.clear()
        self._names.clear()
        for i in range(self.strat_list.count()):
            item = self.strat_list.item(i)
            path = item.data(Qt.UserRole)
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                with open(path, "r", encoding="utf-8") as f:
                    steps = json.load(f)
            except Exception:
                continue
            pred, prec, rec = run_strategy(self.stack_path, self.gt_path, steps)
            self._preds.append(pred)
            self._metrics.append((prec, rec))
            self._names.append(name)
        if self._stack is not None:
            n = self._stack.shape[0]
            self.slice_slider.setRange(0, n - 1)
            self.slice_slider.setValue(0)
            self.slice_slider.setEnabled(True)
        self._build_panels()
        self._update_images()

    def _build_panels(self) -> None:
        # clear layout
        while self.img_layout.count():
            item = self.img_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.labels: List[QLabel] = []
        titles = ["Ground Truth"] + self._names
        for title in titles:
            v = QVBoxLayout()
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            v.addWidget(lbl)
            v.addWidget(QLabel(title))
            self.img_layout.addLayout(v)
            self.labels.append(lbl)

    def _update_images(self) -> None:
        if self._stack is None or self._gt is None:
            return
        idx = self.slice_slider.value()
        img = self._stack[idx]
        gt_slice = self._gt[idx]
        # first panel: GT overlay
        base = overlay_image(img, gt_slice, gt_slice)
        pix = self._to_pixmap(base)
        self.labels[0].setPixmap(pix)
        for i, pred in enumerate(self._preds, start=1):
            overlay = overlay_image(img, gt_slice, pred[idx])
            pix = self._to_pixmap(overlay)
            self.labels[i].setPixmap(pix)
            prec, rec = self._metrics[i - 1]
            self.labels[i].setToolTip(f"precision={prec:.3f} recall={rec:.3f}")

    @staticmethod
    def _to_pixmap(arr: np.ndarray) -> QPixmap:
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, QImage.Format_RGB888)
        return QPixmap.fromImage(img)
