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
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QSpinBox,
    QScrollArea,
    QWidget,
    QGridLayout,
    QInputDialog,
    QMenu,
)
from PyQt5.QtGui import QTransform
from PyQt5.QtCore import Qt

from ..pipeline import (
    run_strategy,
    overlay_mask,
    read_stack,
    StrategyRunner,
    grid_search_strategy,
)
from ..views.canvas import SyncCanvas
from ..models.zstack_model import ZStackModel


class ComparisonDialog(QDialog):
    """Visualize segmentation strategies against ground truth."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Strategy Comparison")
        # Paths and loaded data for multiple stacks
        self.stack_paths: List[str] = []
        self.gt_paths: List[Optional[str]] = []
        self.stacks: List[np.ndarray] = []
        self.gts: List[Optional[np.ndarray]] = []

        self.current_stack: int = 0

        # per-stack predictions (filled when running strategies)
        self._preds_by_stack: List[Optional[List[np.ndarray]]] = []

        # currently displayed data
        self._preds: List[np.ndarray] = []
        self._names: List[str] = []
        self._stack: Optional[np.ndarray] = None
        self._gt: Optional[np.ndarray] = None

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
        self.strat_widget = QWidget()
        self.strat_widget.setLayout(strat_layout)
        layout.addWidget(self.strat_widget)

        # grid search widgets
        self.grid_mode = QCheckBox("Grid Search")
        self.grid_mode.stateChanged.connect(self._toggle_mode)
        layout.addWidget(self.grid_mode)

        grid_layout = QHBoxLayout()
        self.grid_strat_edit = QLineEdit()
        grid_btn = QPushButton("Strategy…")
        grid_btn.clicked.connect(self._choose_grid_strategy)
        grid_layout.addWidget(QLabel("Strategy:"))
        grid_layout.addWidget(self.grid_strat_edit)
        grid_layout.addWidget(grid_btn)
        grid_layout.addWidget(QLabel("Grid:"))
        self.grid_param_edit = QLineEdit()
        self.grid_param_edit.setPlaceholderText("1.value:10,20;2.iter:1,2")
        grid_layout.addWidget(self.grid_param_edit)
        self.grid_widget = QWidget()
        self.grid_widget.setLayout(grid_layout)
        layout.addWidget(self.grid_widget)
        self.grid_widget.hide()

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

        # enable context menu for switching stacks
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    # ---- UI helpers ----
    def _choose_stack(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Stack", "", "TIFF (*.tif *.tiff *.ome.tif)")
        if not paths:
            return

        self.stack_paths = []
        self.gt_paths = []
        self.stacks = []
        self.gts = []
        self._preds_by_stack = []

        for p in paths:
            self.stack_paths.append(p)
            stack = read_stack(p)
            self.stacks.append(stack)
            gt_path, _ = QFileDialog.getOpenFileName(
                self,
                f"Ground Truth for {os.path.basename(p)}",
                "",
                "TIFF (*.tif *.tiff *.ome.tif)",
            )
            if gt_path:
                self.gt_paths.append(gt_path)
                self.gts.append(read_stack(gt_path).astype(np.uint8))
            else:
                self.gt_paths.append(None)
                self.gts.append(None)
            self._preds_by_stack.append(None)

        self.current_stack = 0
        self.stack_edit.setText(self.stack_paths[0])
        self.gt_edit.setText(self.gt_paths[0] or "")
        self._stack = self.stacks[0]
        self._gt = self.gts[0]

        n = self._stack.shape[0]
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, n - 1)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)
        self.slice_slider.setEnabled(True)
        self._preds = []
        self._names.clear()
        self._build_panels()
        self._update_images()

    def _show_context_menu(self, pos) -> None:
        if len(self.stack_paths) <= 1:
            return
        menu = QMenu(self)
        stack_menu = menu.addMenu("Stacks")
        for i, p in enumerate(self.stack_paths):
            act = stack_menu.addAction(os.path.basename(p))
            act.setData(i)
            if i == self.current_stack:
                act.setCheckable(True)
                act.setChecked(True)
        chosen = menu.exec_(self.mapToGlobal(pos))
        if chosen and chosen.data() is not None:
            idx = int(chosen.data())
            self._switch_stack(idx)

    def _switch_stack(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.stacks):
            return
        self.current_stack = idx
        self.stack_edit.setText(self.stack_paths[idx])
        self.gt_edit.setText(self.gt_paths[idx] or "")
        self._load_preview()

    def _choose_gt(self) -> None:
        if not self.stack_paths:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth", "", "TIFF (*.tif *.tiff *.ome.tif)")
        if path:
            self.gt_paths[self.current_stack] = path
            self.gts[self.current_stack] = read_stack(path).astype(np.uint8)
            self.gt_edit.setText(path)
        else:
            self.gt_paths[self.current_stack] = None
            self.gts[self.current_stack] = None
            self.gt_edit.setText("")
        self._gt = self.gts[self.current_stack]
        # invalidate predictions for this stack
        self._preds_by_stack[self.current_stack] = None
        self._preds = []
        self._build_panels()
        self._update_images()

    def _add_strategy(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Strategy", "", "JSON (*.json)")
        if path:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            self.strat_list.addItem(item)
            # invalidate all stored predictions
            self._preds_by_stack = [None for _ in self._preds_by_stack]

    def _remove_strategy(self) -> None:
        row = self.strat_list.currentRow()
        if row >= 0:
            self.strat_list.takeItem(row)
            # invalidate stored predictions for all stacks
            self._preds_by_stack = [None for _ in self._preds_by_stack]

    def _choose_grid_strategy(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Strategy", "", "JSON (*.json)")
        if path:
            self.grid_strat_edit.setText(path)

    def _toggle_mode(self) -> None:
        grid = self.grid_mode.isChecked()
        self.strat_widget.setVisible(not grid)
        self.grid_widget.setVisible(grid)

    def _parse_grid(self, text: str) -> dict[str, list]:
        grid: dict[str, list] = {}
        for part in text.split(";"):
            part = part.strip()
            if not part or ":" not in part:
                continue
            key, values = part.split(":", 1)
            vals: list = []
            for v in values.split(","):
                v = v.strip()
                if not v:
                    continue
                try:
                    if "." in v:
                        vals.append(float(v))
                    else:
                        vals.append(int(v))
                except ValueError:
                    vals.append(v)
            grid[key.strip()] = vals
        return grid

    def _load_preview(self) -> None:
        """Load current stack and ground truth for preview."""
        if not self.stacks:
            return
        self._stack = self.stacks[self.current_stack]
        self._gt = self.gts[self.current_stack]
        self._preds = self._preds_by_stack[self.current_stack] or []
        n = self._stack.shape[0]
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, n - 1)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)
        self.slice_slider.setEnabled(True)
        self._build_panels()

    # ---- Processing ----
    def _run(self) -> None:
        if not self.stacks:
            return
        stack_path = self.stack_paths[self.current_stack]
        gt_path = self.gt_paths[self.current_stack]
        self._stack = self.stacks[self.current_stack]
        self._gt = self.gts[self.current_stack]
        self._preds = []
        self._names.clear()

        slice_idx = self.slice_slider.value() if self.slice_radio.isChecked() else None

        if self.grid_mode.isChecked():
            path = self.grid_strat_edit.text()
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    base_steps = json.load(f)
            except Exception:
                return
            grid = self._parse_grid(self.grid_param_edit.text())
            results = grid_search_strategy(
                stack_path,
                gt_path,
                base_steps,
                grid,
                base_name=os.path.splitext(os.path.basename(path))[0],
                save_dir=os.path.dirname(path),
                slice_idx=slice_idx,
            )
            for name, pred, _p, _r in results:
                self._names.append(name)
                self._preds.append(pred)
            self._build_panels()
        else:
            paths: list[str] = []
            for i in range(self.strat_list.count()):
                item = self.strat_list.item(i)
                path = item.data(Qt.UserRole)
                if path:
                    paths.append(path)
                    name = os.path.splitext(os.path.basename(path))[0]
                    self._names.append(name)

            self._build_panels()

            for si, path in enumerate(paths):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        steps = json.load(f)
                except Exception:
                    continue

                if gt_path:
                    pred, _, _ = run_strategy(
                        stack_path,
                        gt_path,
                        steps,
                        slice_idx=slice_idx,
                        step_callback=lambda m, sidx, step_idx, s=si: self._live_update(s, m, sidx),
                    )
                else:
                    model = ZStackModel()
                    model.load(stack_path)
                    model.ensure_masks()
                    runner = StrategyRunner(model)

                    indices = range(model.n_slices) if slice_idx is None else [slice_idx]
                    for idx in indices:
                        model.index = idx
                        def cb(mask, cur_idx, step_idx, s=si):
                            self._live_update(s, mask, cur_idx)
                        runner.run_steps(steps, callback=cb)
                        model.index = min(model.index, model.n_slices - 1)
                    pred = model.masks.astype(np.uint8)

                self._preds.append(pred)

                base = os.path.splitext(os.path.basename(stack_path))[0]
                if slice_idx is None:
                    default = os.path.join(os.path.dirname(stack_path), f"{base}_{self._names[si]}.tif")
                else:
                    default = os.path.join(
                        os.path.dirname(stack_path), f"{base}_slice{slice_idx + 1}_{self._names[si]}.tif"
                    )
                save_path, _ = QFileDialog.getSaveFileName(self, "Save Result", default, "TIFF Images (*.tif)")
                if save_path:
                    if slice_idx is None:
                        tifffile.imwrite(save_path, pred)
                    else:
                        tifffile.imwrite(save_path, pred[slice_idx])

        # store predictions for this stack
        self._preds_by_stack[self.current_stack] = self._preds

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

        # account for raw and raw+gt windows
        needed = len(self._names) + 2

        # if the requested window count is less than the number of images to
        # display, ask the user which strategies to hide until it fits
        while needed > count:
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
                count = needed
                self.window_spin.setValue(count)
                break
            idx = self._names.index(choice)
            del self._names[idx]
            del self._preds[idx]
            needed = len(self._names) + 2

        count = max(count, needed)
        self.canvases: List[SyncCanvas] = []
        titles = ["Raw", "Raw + GT"] + self._names + [""] * (count - needed)

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
            if i == 0:
                canvas.set_image(
                    np.stack([ZStackModel._normalize_to_8bit(img)] * 3, axis=-1)
                )
            elif i == 1:
                if self._gt is not None:
                    overlay = overlay_mask(img, self._gt[idx], alpha=0.4)
                    canvas.set_image(overlay)
                else:
                    canvas.set_image(
                        np.stack([ZStackModel._normalize_to_8bit(img)] * 3, axis=-1)
                    )
            else:
                p_idx = i - 2
                if p_idx < len(self._preds):
                    overlay = overlay_mask(img, self._preds[p_idx][idx], alpha=0.4)
                    canvas.set_image(overlay)
                else:
                    h, w = img.shape
                    canvas.set_image(np.zeros((h, w, 3), dtype=np.uint8))
            canvas.setToolTip("")

    def _live_update(self, strat_idx: int, masks: np.ndarray, slice_idx: int) -> None:
        """Update a canvas during strategy execution."""
        if self._stack is None or strat_idx + 2 >= len(self.canvases):
            return
        if strat_idx < len(self._preds):
            self._preds[strat_idx] = masks
        else:
            # when called before final append
            self._preds.append(masks)
        if slice_idx == self.slice_slider.value():
            img = self._stack[slice_idx]
            overlay = overlay_mask(img, masks[slice_idx], alpha=0.4)
            self.canvases[strat_idx + 2].set_image(overlay)

