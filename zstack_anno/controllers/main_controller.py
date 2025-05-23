from PyQt5.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QSlider,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QEvent, QPoint
import sys
import numpy as np
from ..models.zstack_model import ZStackModel
from ..views.canvas import SliceCanvas
from ..utils import morphology_tools

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z-Stack Annotation (alpha)")
        self.model = ZStackModel()
        self.canvas = SliceCanvas()
        self.undo_stack: list[tuple[np.ndarray, str]] = []
        self.redo_stack: list[tuple[np.ndarray, str]] = []
        self.history: list[str] = []
        # Brush tool settings
        self.brush_enabled: bool = False
        self.brush_size: int = 5
        self._painting: bool = False
        self._last_pos = None
        self._temp_mask = None
        self._build_layout()
        self._create_menu()
        self.statusBar().showMessage("Ready")
        # Capture key events from child widgets
        self.installEventFilter(self)
        self.canvas.installEventFilter(self)
        # Also filter events from the canvas viewport for painting
        self.canvas.viewport().installEventFilter(self)
        self.slider.installEventFilter(self)

    def _build_layout(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        # -------- Controls --------
        ctrl = QHBoxLayout()
        self.prev_btn = QPushButton("Prev")
        self.prev_btn.clicked.connect(self._prev_slice)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self._next_slice)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slice_changed)
        ctrl.addWidget(self.prev_btn)
        ctrl.addWidget(self.slider)
        ctrl.addWidget(self.next_btn)

        self.dilate_btn = QPushButton("Dilate")
        self.dilate_btn.clicked.connect(self._dilate_current)
        self.erode_btn = QPushButton("Erode")
        self.erode_btn.clicked.connect(self._erode_current)
        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(1, 10)
        self.strength_spin.setValue(1)
        ctrl.addWidget(self.dilate_btn)
        ctrl.addWidget(self.erode_btn)
        ctrl.addWidget(self.strength_spin)

        self.filter_btn = QPushButton("Filter <")
        self.filter_btn.clicked.connect(self._filter_small)
        self.filter_spin = QSpinBox()
        self.filter_spin.setRange(1, 10000)
        self.filter_spin.setValue(100)
        ctrl.addWidget(self.filter_btn)
        ctrl.addWidget(self.filter_spin)

        self.info_label = QLabel("")
        ctrl.addWidget(self.info_label)
        layout.addLayout(ctrl)
        self.setCentralWidget(central)

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("File")
        open_act = file_menu.addAction("Open…")
        open_act.triggered.connect(self._open_file)
        open_act.setShortcuts(["Ctrl+O", "Meta+O"])

        new_mask_act = file_menu.addAction("New Mask Stack…")
        new_mask_act.triggered.connect(self._create_masks)
        new_mask_act.setShortcuts(["Ctrl+M", "Meta+M"])

        open_mask_act = file_menu.addAction("Open Masks…")
        open_mask_act.triggered.connect(self._open_masks)
        open_mask_act.setShortcuts(["Ctrl+Shift+M", "Meta+Shift+M"])
        save_mask_act = file_menu.addAction("Save Masks…")
        save_mask_act.triggered.connect(self._save_masks)

        edit_menu = self.menuBar().addMenu("Edit")
        dilate_act = edit_menu.addAction("Dilate")
        dilate_act.triggered.connect(self._dilate_current)
        erode_act = edit_menu.addAction("Erode")
        erode_act.triggered.connect(self._erode_current)
        edit_menu.addSeparator()
        undo_act = edit_menu.addAction("Undo")
        undo_act.triggered.connect(self._undo)
        redo_act = edit_menu.addAction("Redo")
        redo_act.triggered.connect(self._redo)

    def _open_file(self):
        if not self._prompt_save_if_dirty():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF", "", "TIFF Images (*.tif *.tiff *.ome.tif)")
        if path:
            self.model.load(path)
            self.slider.setRange(0, self.model.n_slices - 1)
            self.slider.setEnabled(True)
            self._update_view(reset_view=True)

    def _open_masks(self):
        if not self._prompt_save_if_dirty():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Mask Stack", "", "TIFF Images (*.tif *.tiff)")
        if path:
            self.model.load_masks(path)
            self._update_view()

    def _save_masks(self):
        if self.model.masks is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Masks",
            self.model.mask_path or (self.model.default_mask_path() if self.model.data is not None else ""),
            "TIFF Images (*.tif)",
        )
        if path:
            self.model.save_masks(path)

    def _create_masks(self):
        if self.model.data is None:
            return
        default = self.model.default_mask_path()
        path, _ = QFileDialog.getSaveFileName(
            self, "Create Mask Stack", default, "TIFF Images (*.tif)"
        )
        if path:
            self.model.create_blank_masks(path)
            self._update_view()

    def _prompt_save_if_dirty(self) -> bool:
        if not self.model.mask_dirty:
            return True
        ret = QMessageBox.question(
            self,
            "Save Masks",
            "Save mask changes?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if ret == QMessageBox.Cancel:
            return False
        if ret == QMessageBox.Yes:
            if self.model.mask_path is None:
                self._save_masks()
            else:
                self.model.save_masks()
        return True

    def closeEvent(self, event):
        if self._prompt_save_if_dirty():
            super().closeEvent(event)
        else:
            event.ignore()

    def _on_slice_changed(self, idx: int):
        if self.model.data is None:
            return
        self.model.index = idx
        self._update_view()

    def _prev_slice(self):
        if not self.slider.isEnabled():
            return
        self.slider.setValue(max(0, self.slider.value() - 1))

    def _next_slice(self):
        if not self.slider.isEnabled():
            return
        self.slider.setValue(min(self.model.n_slices - 1, self.slider.value() + 1))

    def _update_view(self, reset_view: bool = False):
        self.canvas.set_image(self.model.get_current(), reset_view=reset_view)
        mask = self.model.get_mask() if self.model.masks is not None else None
        self.canvas.set_mask(mask)
        self.statusBar().showMessage(
            f"Slice {self.model.index + 1} / {self.model.n_slices}")
        info = (
            f"Components: {self.model.component_count()}  "
            f"Pixels: {self.model.total_pixel_count()}"
        )
        if hasattr(self, "info_label"):
            self.info_label.setText(info)

    # --------- 辅助函数 ---------
    def _ensure_masks(self) -> bool:
        """确保模型中存在掩膜栈，若无则创建全零栈。"""
        if self.model.data is None:
            return False
        if self.model.masks is None:
            self.model.create_blank_masks()
        return True

    def _push_undo(self, action: str = "") -> None:
        if self.model.masks is None:
            return
        self.undo_stack.append((self.model.masks.copy(), action))
        self.history.append(action)
        if len(self.undo_stack) > 5:
            self.undo_stack.pop(0)
            if self.history:
                self.history.pop(0)
        self.redo_stack.clear()

    # --------- 文件操作 ---------

    # 键盘快捷
    def _handle_key(self, event):
        if not self.slider.isEnabled():
            return
        if event.key() in (Qt.Key_Up, Qt.Key_Left, Qt.Key_W, Qt.Key_A):
            self.slider.setValue(max(0, self.slider.value() - 1))
        elif event.key() in (Qt.Key_Down, Qt.Key_Right, Qt.Key_S):
            self.slider.setValue(min(self.model.n_slices - 1, self.slider.value() + 1))
        elif event.key() == Qt.Key_D:
            self._dilate_current()
        elif event.key() == Qt.Key_E:
            self._erode_current()
        elif event.key() == Qt.Key_Z:
            self._undo()
        elif event.key() == Qt.Key_X:
            self._redo()
        elif event.key() == Qt.Key_P:
            self.brush_enabled = not self.brush_enabled
            self.statusBar().showMessage(
                "Brush ON" if self.brush_enabled else "Brush OFF")
        elif event.key() == Qt.Key_BracketLeft:
            self.brush_size = max(1, self.brush_size - 1)
        elif event.key() == Qt.Key_BracketRight:
            self.brush_size += 1

    # --------- 形态学与撤销重做 ---------
    def _dilate_current(self) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("dilate")
        cur = self.model.get_mask()
        iterations = self.strength_spin.value() if hasattr(self, "strength_spin") else 1
        new = morphology_tools.dilate(cur, iterations=iterations)
        self.model.set_mask(new)
        self._update_view()

    def _erode_current(self) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("erode")
        cur = self.model.get_mask()
        iterations = self.strength_spin.value() if hasattr(self, "strength_spin") else 1
        new = morphology_tools.erode(cur, iterations=iterations)
        self.model.set_mask(new)
        self._update_view()

    def _filter_small(self) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("filter")
        cur = self.model.get_mask()
        thresh = self.filter_spin.value() if hasattr(self, "filter_spin") else 100
        new = morphology_tools.remove_small(cur, thresh)
        self.model.set_mask(new)
        self._update_view()

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        last_mask, action = self.undo_stack.pop()
        self.redo_stack.append((self.model.masks.copy(), action))
        self.model.masks = last_mask
        if self.history:
            self.history.pop()
        self._update_view()

    def _redo(self) -> None:
        if not self.redo_stack:
            return
        last_mask, action = self.redo_stack.pop()
        self.undo_stack.append((self.model.masks.copy(), action))
        self.model.masks = last_mask
        self.history.append(action)
        self._update_view()

    # --------- event filter ---------
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            self._handle_key(event)
            return True
        if (
            self.brush_enabled
            and obj in (self.canvas, self.canvas.viewport())
            and event.type() in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease)
        ):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._start_paint(event.pos())
                return True
            if event.type() == QEvent.MouseMove and event.buttons() & Qt.LeftButton:
                self._continue_paint(event.pos())
                return True
            if event.type() == QEvent.MouseButtonRelease:
                self._end_paint()
                return True
        return super().eventFilter(obj, event)

    # --------- painting helpers ---------
    def _paint_at(self, pos) -> None:
        if not self._ensure_masks():
            return
        scene_pos = self.canvas.mapToScene(pos)
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        if self._temp_mask is None:
            self._temp_mask = self.model.get_mask().copy()
        mask = self._temp_mask
        half = self.brush_size // 2
        y0 = max(0, y - half)
        y1 = min(mask.shape[0], y + half + 1)
        x0 = max(0, x - half)
        x1 = min(mask.shape[1], x + half + 1)
        mask[y0:y1, x0:x1] = 1
        self.canvas.set_mask(mask)

    def _start_paint(self, pos) -> None:
        if self._painting:
            return
        self._push_undo("paint")
        self._painting = True
        self._last_pos = pos
        self._temp_mask = self.model.get_mask().copy()
        self._paint_at(pos)

    def _continue_paint(self, pos) -> None:
        if not self._painting:
            return
        if self._last_pos is None:
            self._last_pos = pos
        self._paint_line(self._last_pos, pos)
        self._last_pos = pos

    def _end_paint(self) -> None:
        if not self._painting:
            return
        self._painting = False
        if self._temp_mask is not None:
            self.model.set_mask(self._temp_mask)
        self._temp_mask = None
        self._last_pos = None
        self._update_view()

    def _paint_line(self, start, end) -> None:
        """Interpolate between points to draw a continuous line."""
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        steps = int(max(abs(dx), abs(dy)))
        if steps == 0:
            self._paint_at(start)
            return
        for i in range(steps + 1):
            x = int(round(start.x() + dx * i / steps))
            y = int(round(start.y() + dy * i / steps))
            self._paint_at(QPoint(x, y))

