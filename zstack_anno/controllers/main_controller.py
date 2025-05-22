from PyQt5.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QSlider,
    QWidget,
    QVBoxLayout,
)
from PyQt5.QtCore import Qt
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
        self.undo_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []
        self._build_layout()
        self._create_menu()
        self.statusBar().showMessage("Ready")

    def _build_layout(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        # -------- Slider --------
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slice_changed)
        layout.addWidget(self.slider)
        self.setCentralWidget(central)

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("File")
        open_act = file_menu.addAction("Open…")
        open_act.triggered.connect(self._open_file)

        open_mask_act = file_menu.addAction("Open Masks…")
        open_mask_act.triggered.connect(self._open_masks)
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF", "", "TIFF Images (*.tif *.tiff *.ome.tif)")
        if path:
            self.model.load(path)
            self.slider.setRange(0, self.model.n_slices - 1)
            self.slider.setEnabled(True)
            self._update_view(reset_view=True)

    def _open_masks(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Mask Stack", "", "TIFF Images (*.tif *.tiff)")
        if path:
            self.model.load_masks(path)
            self._update_view()

    def _save_masks(self):
        if self.model.masks is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Masks", "", "TIFF Images (*.tif)")
        if path:
            self.model.save_masks(path)

    def _on_slice_changed(self, idx: int):
        if self.model.data is None:
            return
        self.model.index = idx
        self._update_view()

    def _update_view(self, reset_view: bool = False):
        self.canvas.set_image(self.model.get_current(), reset_view=reset_view)
        mask = self.model.get_mask() if self.model.masks is not None else None
        self.canvas.set_mask(mask)
        self.statusBar().showMessage(
            f"Slice {self.model.index + 1} / {self.model.n_slices}")

    # --------- 辅助函数 ---------
    def _ensure_masks(self) -> bool:
        """确保模型中存在掩膜栈，若无则创建全零栈。"""
        if self.model.data is None:
            return False
        if self.model.masks is None:
            self.model.masks = np.zeros_like(self.model.data, dtype=np.uint8)
        return True

    def _push_undo(self) -> None:
        if self.model.masks is None:
            return
        self.undo_stack.append(self.model.masks.copy())
        if len(self.undo_stack) > 10:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    # --------- 文件操作 ---------

    # 键盘快捷
    def keyPressEvent(self, event):
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
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self._undo()
        elif event.key() == Qt.Key_Y and event.modifiers() & Qt.ControlModifier:
            self._redo()

    # --------- 形态学与撤销重做 ---------
    def _dilate_current(self) -> None:
        if not self._ensure_masks():
            return
        self._push_undo()
        cur = self.model.get_mask()
        new = morphology_tools.dilate(cur)
        self.model.set_mask(new)
        self._update_view()

    def _erode_current(self) -> None:
        if not self._ensure_masks():
            return
        self._push_undo()
        cur = self.model.get_mask()
        new = morphology_tools.erode(cur)
        self.model.set_mask(new)
        self._update_view()

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(self.model.masks.copy())
        self.model.masks = self.undo_stack.pop()
        self._update_view()

    def _redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(self.model.masks.copy())
        self.model.masks = self.redo_stack.pop()
        self._update_view()

