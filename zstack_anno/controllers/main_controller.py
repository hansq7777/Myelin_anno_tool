from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QSlider,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QLabel,
    QSpinBox,
    QInputDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QEvent, QPoint, QThread, pyqtSignal
import threading
import sys
import os
import numpy as np
from ..models.zstack_model import ZStackModel
from ..views.canvas import SliceCanvas
from ..views.script_editor import ScriptEditor
from ..utils import morphology_tools
from ..utils.dialogs import question_with_shortcuts
from ..utils import config


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
        if mask is None:
            return
        if total == 0:
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
        self._delete_start = None
        self.cancel_event = threading.Event()
        self.grow_thread: IntGrowThread | None = None
        self.script_editor: ScriptEditor | None = None
        self._build_layout()
        self._create_menu()
        self.statusBar().showMessage("Ready")
        # Show image and mask path in the status bar
        self.image_label = QLabel("")
        self.mask_label = QLabel("")
        self.statusBar().addPermanentWidget(self.image_label)
        self.statusBar().addPermanentWidget(self.mask_label)
        self._update_file_labels()
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
        self.title_label = QLabel("Z-Stack Annotation")
        self.title_label.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self.title_label)

        # Navigation controls placed prominently below the title
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Prev")
        self.prev_btn.clicked.connect(self._prev_slice)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slice_changed)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self._next_slice)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.slider)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)

        layout.addWidget(self.canvas)
        # -------- Controls --------
        ctrl = QHBoxLayout()

        morph_layout = QVBoxLayout()
        self.dilate_btn = QPushButton("Dilate")
        self.dilate_btn.clicked.connect(self._dilate_current)
        self.erode_btn = QPushButton("Erode")
        self.erode_btn.clicked.connect(self._erode_current)
        self.skeleton_btn = QPushButton("Skeleton")
        self.skeleton_btn.clicked.connect(self._skeletonize_current)
        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(1, 10)
        self.strength_spin.setValue(1)
        morph_layout.addWidget(self.dilate_btn)
        morph_layout.addWidget(self.erode_btn)
        morph_layout.addWidget(self.skeleton_btn)
        morph_layout.addWidget(self.strength_spin)
        ctrl.addLayout(morph_layout)

        filter_layout = QVBoxLayout()
        self.filter_btn = QPushButton("Filter <")
        self.filter_btn.clicked.connect(self._filter_small)
        self.filter_spin = QSpinBox()
        self.filter_spin.setRange(1, 10000)
        self.filter_spin.setValue(100)
        filter_layout.addWidget(self.filter_btn)
        filter_layout.addWidget(self.filter_spin)
        ctrl.addLayout(filter_layout)

        thresh_layout = QVBoxLayout()
        self.abs_thresh_edit = QLineEdit()
        self.abs_thresh_edit.setPlaceholderText("Abs >")
        self.abs_thresh_btn = QPushButton("Th Abs")
        self.abs_thresh_btn.clicked.connect(self._threshold_abs)
        self.norm_thresh_edit = QLineEdit()
        self.norm_thresh_edit.setPlaceholderText("Norm %")
        self.norm_thresh_btn = QPushButton("Th Norm")
        self.norm_thresh_btn.clicked.connect(self._threshold_norm)
        thresh_layout.addWidget(self.abs_thresh_edit)
        thresh_layout.addWidget(self.abs_thresh_btn)
        thresh_layout.addWidget(self.norm_thresh_edit)
        thresh_layout.addWidget(self.norm_thresh_btn)
        ctrl.addLayout(thresh_layout)


        bg_layout = QVBoxLayout()
        self.bg_percentile_edit = QLineEdit()
        self.bg_percentile_edit.setPlaceholderText("BG %")
        self.bg_bins_edit = QLineEdit()
        self.bg_bins_edit.setPlaceholderText("Bins")
        self.bg_filter_button = QPushButton("BG Filter")
        self.bg_filter_button.clicked.connect(self._apply_bg_filter)
        bg_layout.addWidget(self.bg_percentile_edit)
        bg_layout.addWidget(self.bg_bins_edit)
        bg_layout.addWidget(self.bg_filter_button)
        ctrl.addLayout(bg_layout)

        stretch_layout = QVBoxLayout()
        self.stretch_edit = QLineEdit()
        self.stretch_edit.setPlaceholderText("Stretch %")
        self.stretch_button = QPushButton("Stretch")
        self.stretch_button.clicked.connect(self._apply_stretch)
        stretch_layout.addWidget(self.stretch_edit)
        stretch_layout.addWidget(self.stretch_button)
        ctrl.addLayout(stretch_layout)

        blur_layout = QVBoxLayout()
        self.blur_btn = QPushButton("Blur")
        self.blur_btn.clicked.connect(self._apply_blur)
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(1, 10)
        self.blur_spin.setValue(1)
        self.show_orig_chk = QCheckBox("Show Original")
        self.show_orig_chk.toggled.connect(self._toggle_original)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._change_opacity)
        self.clear_blur_btn = QPushButton("Clear Blur")
        self.clear_blur_btn.clicked.connect(self._clear_blur)
        blur_layout.addWidget(self.blur_btn)
        blur_layout.addWidget(self.blur_spin)
        blur_layout.addWidget(self.show_orig_chk)
        blur_layout.addWidget(self.opacity_slider)
        blur_layout.addWidget(self.clear_blur_btn)
        ctrl.addLayout(blur_layout)

        grow_layout = QVBoxLayout()
        self.seed_thresh_edit = QLineEdit()
        self.seed_thresh_edit.setPlaceholderText("Seed %")
        self.seed_btn = QPushButton("Seed")
        self.seed_btn.clicked.connect(self._seed_current)
        self.int_diff_edit = QLineEdit()
        self.int_diff_edit.setPlaceholderText("Diff %")
        self.int_hist_edit = QLineEdit()
        self.int_hist_edit.setPlaceholderText("Hist %")
        self.int_grow_btn = QPushButton("Int Grow")
        self.int_grow_btn.clicked.connect(self._grow_intensity)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_operation)
        self.cancel_btn.setEnabled(False)
        grow_layout.addWidget(self.seed_thresh_edit)
        grow_layout.addWidget(self.seed_btn)
        grow_layout.addWidget(self.int_diff_edit)
        grow_layout.addWidget(self.int_hist_edit)
        grow_layout.addWidget(self.int_grow_btn)
        grow_layout.addWidget(self.cancel_btn)
        ctrl.addLayout(grow_layout)

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
        new_mask_act.setShortcuts(["Ctrl+Shift+M", "Meta+Shift+M"])

        open_mask_act = file_menu.addAction("Open Masks…")
        open_mask_act.triggered.connect(self._open_masks)
        open_mask_act.setShortcuts(["Ctrl+M", "Meta+M"])

        mask_folder_act = file_menu.addAction("Set Mask Folder…")
        mask_folder_act.triggered.connect(self._set_mask_folder)

        quick_save_act = file_menu.addAction("Quick Save")
        quick_save_act.triggered.connect(self._quick_save_masks)
        # support Command+S on macOS while keeping the existing Option+S
        # shortcut. Ctrl+S is also added for consistency on other platforms.
        quick_save_act.setShortcuts(["Alt+S", "Ctrl+S", "Meta+S"])

        save_mask_act = file_menu.addAction("Save Masks…")
        save_mask_act.triggered.connect(self._save_masks)

        edit_menu = self.menuBar().addMenu("Edit")
        undo_act = edit_menu.addAction("Undo")
        undo_act.triggered.connect(self._undo)
        redo_act = edit_menu.addAction("Redo")
        redo_act.triggered.connect(self._redo)

        mask_menu = self.menuBar().addMenu("Mask")
        dilate_act = mask_menu.addAction("Dilate")
        dilate_act.triggered.connect(self._dilate_current)
        erode_act = mask_menu.addAction("Erode")
        erode_act.triggered.connect(self._erode_current)
        skel_act = mask_menu.addAction("Skeleton")
        skel_act.triggered.connect(self._skeletonize_current)
        filter_act = mask_menu.addAction("Filter Small")
        filter_act.triggered.connect(self._filter_small)
        thresh_abs_act = mask_menu.addAction("Threshold Abs")
        thresh_abs_act.triggered.connect(self._threshold_abs)
        thresh_norm_act = mask_menu.addAction("Threshold Norm")
        thresh_norm_act.triggered.connect(self._threshold_norm)
        bg_act = mask_menu.addAction("Remove Background")
        bg_act.triggered.connect(self._apply_bg_filter)
        seed_act = mask_menu.addAction("Seed")
        seed_act.triggered.connect(self._seed_current)
        int_grow_act = mask_menu.addAction("Intensity Grow")
        int_grow_act.triggered.connect(self._grow_intensity)

        image_menu = self.menuBar().addMenu("Image")
        stretch_act = image_menu.addAction("Histogram Stretch")
        stretch_act.triggered.connect(self._apply_stretch)
        blur_act = image_menu.addAction("Gaussian Blur")
        blur_act.triggered.connect(self._apply_blur)
        self.show_orig_act = image_menu.addAction("Show Original")
        self.show_orig_act.setCheckable(True)
        self.show_orig_act.toggled.connect(self.show_orig_chk.setChecked)
        self.show_orig_chk.toggled.connect(self.show_orig_act.setChecked)
        self.show_orig_act.setChecked(self.show_orig_chk.isChecked())
        clear_blur_act = image_menu.addAction("Clear Blur")
        clear_blur_act.triggered.connect(self._clear_blur)

        tool_menu = self.menuBar().addMenu("Tools")
        script_act = tool_menu.addAction("Script Editor")
        script_act.triggered.connect(self._open_script_editor)
        # Support Command+E and Option+E on macOS, and Alt+E elsewhere.
        # Ctrl+E is mapped to Command+E automatically on macOS, so include
        # it alongside Alt/Meta for cross-platform compatibility.
        script_act.setShortcuts(["Alt+E", "Ctrl+E", "Meta+E"])

        help_menu = self.menuBar().addMenu("Help")
        help_act = help_menu.addAction("Shortcuts && Features")
        help_act.triggered.connect(self._show_help)

    def _open_file(self):
        if not self._prompt_save_if_dirty():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF", "", "TIFF Images (*.tif *.tiff *.ome.tif)")
        if path:
            self.model.load(path)
            self.slider.setRange(0, self.model.n_slices - 1)
            self.slider.setEnabled(True)
            mask_folder = config.get("mask_folder")
            if mask_folder:
                base = os.path.splitext(os.path.basename(path))[0] + "_mask.tif"
                mask_path = os.path.join(mask_folder, base)
                if os.path.exists(mask_path):
                    try:
                        self.model.load_masks(mask_path)
                    except Exception:
                        pass
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
            self._update_file_labels()

    def _quick_save_masks(self) -> None:
        """Save masks to existing path or prompt for one if needed."""
        if self.model.masks is None:
            return
        if self.model.mask_path is None:
            self._save_masks()
        else:
            self.model.save_masks()
        self._update_file_labels()

    def _create_masks(self):
        if self.model.data is None:
            return
        default_folder = os.path.dirname(self.model.path) if self.model.path else ""
        folder = QFileDialog.getExistingDirectory(
            self, "Create Mask Stack", default_folder
        )
        if folder:
            base = os.path.splitext(os.path.basename(self.model.path))[0] + "_mask.tif"
            path = os.path.join(folder, base)
            self.model.create_blank_masks(path)
            self._update_view()

    def _prompt_save_if_dirty(self) -> bool:
        if not self.model.mask_dirty:
            return True
        ret = question_with_shortcuts(
            self,
            "Save Masks",
            "Save mask changes?",
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
        self._update_file_labels()

    def _progress_update(self, cur: int, total: int, mask: np.ndarray | None = None) -> None:
        if mask is None or total == 0:
            return
        if not hasattr(self, "_next_progress"):
            self._next_progress = 0.2
        frac = cur / float(total)
        if frac >= self._next_progress:
            self.canvas.set_mask(mask)
            QApplication.processEvents()
            self._next_progress += 0.2

    # --------- 辅助函数 ---------
    def _ensure_masks(self) -> bool:
        """Ensure mask stack exists, allocating it in memory if necessary."""
        if self.model.data is None:
            return False
        if self.model.masks is None:
            # Create the mask stack in memory without writing to disk
            self.model.ensure_masks()
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

    def _short_path(self, path: str | None) -> str:
        if not path:
            return "(none)"
        parent = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)
        return os.path.join(parent, name)

    def _update_file_labels(self) -> None:
        if hasattr(self, "image_label"):
            self.image_label.setText(f"Image: {self._short_path(self.model.path)}")
        if hasattr(self, "mask_label"):
            self.mask_label.setText(f"Mask: {self._short_path(self.model.mask_path)}")

    # --------- 文件操作 ---------

    # 键盘快捷
    def _handle_key(self, event):
        if not self.slider.isEnabled():
            return
        if event.key() in (Qt.Key_Up, Qt.Key_Left):
            self.slider.setValue(max(0, self.slider.value() - 1))
        elif event.key() in (Qt.Key_Down, Qt.Key_Right):
            self.slider.setValue(min(self.model.n_slices - 1, self.slider.value() + 1))
        elif (
            event.key() == Qt.Key_S
            and event.modifiers()
            in (Qt.AltModifier, Qt.ControlModifier, Qt.MetaModifier)
        ):
            self._quick_save_masks()
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
        elif event.key() == Qt.Key_H:
            self.brush_enabled = False
            self.canvas.setDragMode(self.canvas.ScrollHandDrag)
            self.statusBar().showMessage("Hand tool")
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

    def _skeletonize_current(self) -> None:
        if not self._ensure_masks():
            return
        algorithms = ["skeletonize", "skeletonize_3d", "medial_axis"]
        alg, ok = QInputDialog.getItem(
            self, "Skeletonize", "Algorithm:", algorithms, 0, False
        )
        if not ok or not alg:
            return
        params: dict[str, object] = {}
        if alg == "medial_axis":
            choice, ok = QInputDialog.getItem(
                self,
                "Return Distance",
                "Return distance?",
                ["False", "True"],
                0,
                False,
            )
            if not ok:
                return
            params["return_distance"] = choice == "True"
        self._push_undo("skeletonize")
        if alg == "skeletonize_3d":
            if self.model.masks is None:
                return
            result = morphology_tools.skeletonize_stack(
                self.model.masks, algorithm=alg, **params
            )
            self.model.masks = result
        else:
            cur = self.model.get_mask()
            new = morphology_tools.skeletonize_slice(cur, algorithm=alg, **params)
            self.model.set_mask(new)
        self._update_view()

    def _threshold_abs(self) -> None:
        if not self._ensure_masks():
            return
        try:
            value = float(self.abs_thresh_edit.text())
        except ValueError:
            value = 0.0
        self._push_undo("thresh_abs")
        self.model.threshold_absolute(value)
        self._update_view()

    def _threshold_norm(self) -> None:
        if not self._ensure_masks():
            return
        try:
            pct = float(self.norm_thresh_edit.text())
        except ValueError:
            pct = 50.0
        self._push_undo("thresh_norm")
        self.model.threshold_normalized(pct)
        self._update_view()

    def _apply_bg_filter(self) -> None:
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
        self.model.remove_background(
            pct, bins, progress=True, progress_fn=self._progress_update
        )
        self._update_view()

    def _apply_stretch(self) -> None:
        if self.model.data is None:
            return
        try:
            pct = float(self.stretch_edit.text())
        except ValueError:
            pct = 0.0
        if pct <= 0:
            self.model.reset_contrast()
        else:
            self.model.histogram_stretch(pct)
        # keep current zoom level when updating the view after stretch
        self._update_view()

    def _apply_blur(self) -> None:
        if self.model.data is None:
            return
        sigma = float(self.blur_spin.value()) if hasattr(self, "blur_spin") else 1.0
        self.model.apply_gaussian_blur(sigma)
        # keep current zoom level when updating the view after blur
        self._update_view()

    def _toggle_original(self) -> None:
        self.model.toggle_show_original()
        # keep current zoom level when toggling original view
        self._update_view()

    def _change_opacity(self) -> None:
        value = (
            self.opacity_slider.value()
            if hasattr(self, "opacity_slider")
            else 50
        )
        self.canvas.set_mask_opacity(value / 100.0)
        # refresh current mask to apply new opacity
        mask = self.model.get_mask() if self.model.masks is not None else None
        self.canvas.set_mask(mask)

    def _clear_blur(self) -> None:
        self.model.remove_gaussian_blur()
        # do not reset the view when clearing blur
        self._update_view()

    def _seed_current(self) -> None:
        if not self._ensure_masks():
            return
        try:
            pct = float(self.seed_thresh_edit.text())
        except ValueError:
            pct = 90.0
        self._push_undo("seed")
        img = self.model.get_current()
        cur = self.model.get_mask()
        seeds = morphology_tools.sample_seeds(img, pct, num_seeds=20000)
        cur = cur.copy()
        cur[seeds > 0] = 1
        self.model.set_mask(cur)
        self._update_view()
    def _grow_intensity(self) -> None:
        if not self._ensure_masks():
            return
        if self.grow_thread is not None and self.grow_thread.isRunning():
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
        img = self.model.get_current()
        cur = self.model.get_mask()
        self.cancel_event.clear()
        self.cancel_btn.setEnabled(True)
        self.int_grow_btn.setEnabled(False)
        self.grow_thread = IntGrowThread(
            img.astype(float), cur, diff_pct, hist_pct, None, self.cancel_event
        )
        self.grow_thread.finished.connect(self._int_grow_finished)
        self.grow_thread.cancelled.connect(self._int_grow_cancelled)
        self.grow_thread.progress.connect(self._thread_progress)
        self.grow_thread.start()

    def _int_grow_finished(self, result: np.ndarray) -> None:
        self.model.set_mask(result)
        self._update_view()
        self.int_grow_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.grow_thread = None

    def _int_grow_cancelled(self) -> None:
        self.int_grow_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.statusBar().showMessage("Operation cancelled")
        self.grow_thread = None

    def _thread_progress(self, mask: np.ndarray, cur: int, total: int) -> None:
        self._progress_update(cur, total, mask)

    def _cancel_operation(self) -> None:
        self.cancel_event.set()

    # --------- scriptable wrappers ---------
    def script_dilate(self, iterations: int = 1) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("dilate")
        cur = self.model.get_mask()
        new = morphology_tools.dilate(cur, iterations=iterations)
        self.model.set_mask(new)
        self._update_view()

    def script_erode(self, iterations: int = 1) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("erode")
        cur = self.model.get_mask()
        new = morphology_tools.erode(cur, iterations=iterations)
        self.model.set_mask(new)
        self._update_view()

    def script_filter_small(self, threshold: int = 100) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("filter")
        cur = self.model.get_mask()
        new = morphology_tools.remove_small(cur, threshold)
        self.model.set_mask(new)
        self._update_view()

    def script_skeletonize(
        self, algorithm: str = "skeletonize", return_distance: bool = False
    ) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("skeletonize")
        if algorithm == "skeletonize_3d":
            if self.model.masks is None:
                return
            result = morphology_tools.skeletonize_stack(
                self.model.masks, algorithm=algorithm
            )
            self.model.masks = result
        else:
            cur = self.model.get_mask()
            params = {}
            if algorithm == "medial_axis":
                params["return_distance"] = return_distance
            new = morphology_tools.skeletonize_slice(cur, algorithm=algorithm, **params)
            self.model.set_mask(new)
        self._update_view()

    def script_threshold_abs(self, value: float = 0.0) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("thresh_abs")
        self.model.threshold_absolute(value)
        self._update_view()

    def script_threshold_norm(self, percent: float = 50.0) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("thresh_norm")
        self.model.threshold_normalized(percent)
        self._update_view()

    def script_seed(self, percentile: float = 90.0) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("seed")
        img = self.model.get_current()
        cur = self.model.get_mask()
        seeds = morphology_tools.sample_seeds(img, percentile, num_seeds=20000)
        cur = cur.copy()
        cur[seeds > 0] = 1
        self.model.set_mask(cur)
        self._update_view()

    def script_int_grow(
        self,
        diff_pct: float = 20.0,
        hist_pct: float | None = None,
        force_pct: float | None = None,
        limit: int | None = 30000,
    ) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("int_grow")
        img = self.model.get_current()
        cur = self.model.get_mask()
        self._next_progress = 0.2
        grown = morphology_tools.intensity_region_grow(
            img.astype(float),
            cur,
            diff_pct,
            hist_pct,
            force_pct,
            limit,
            progress=True,
            progress_fn=self._progress_update,
        )
        self.model.set_mask(grown)
        self._update_view()

    def script_flood_grow(
        self, connectivity: int = 1, tolerance: float = 5.0, workers: int = 1
    ) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("flood_grow")
        img = self.model.get_current()
        cur = self.model.get_mask()
        self._next_progress = 0.2
        grown = morphology_tools.flood_region_grow(
            img.astype(float),
            cur,
            connectivity=connectivity,
            tolerance=tolerance,
            workers=workers,
            progress=True,
            progress_fn=self._progress_update,
        )
        self.model.set_mask(grown)
        self._update_view()

    def script_blur(self, sigma: float = 1.0) -> None:
        if self.model.data is None:
            return
        self.model.apply_gaussian_blur(sigma)
        self._update_view()

    def script_clear_blur(self) -> None:
        self.model.remove_gaussian_blur()
        self._update_view()

    def script_bg_filter(self, percentile: float = 0.0, bins: int = 0) -> None:
        """Remove low intensity pixels from the mask using percentile."""
        if not self._ensure_masks():
            return
        self._push_undo("bg_filter")
        self._next_progress = 0.2
        self.model.remove_background(
            percentile, bins, progress=True, progress_fn=self._progress_update
        )
        self._update_view()

    def script_next_slice(self) -> None:
        self._next_slice()

    def script_prev_slice(self) -> None:
        self._prev_slice()

    def script_save(self) -> None:
        """Save current masks using the quick save logic."""
        self._quick_save_masks()

    def report_action(self, action: str, params: dict) -> None:
        path = os.path.basename(self.model.path or "")
        msg_params = ", ".join(f"{k}={v}" for k, v in params.items())
        print(
            f"{path} slice {self.model.index + 1}/{self.model.n_slices}: {action} {msg_params}"
        )

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
        if obj in (self.canvas, self.canvas.viewport()):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                self._delete_start = event.pos()
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
                if self._delete_start is not None:
                    self._delete_rect(self._delete_start, event.pos())
                    self._delete_start = None
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

    def _delete_rect(self, start, end) -> None:
        """Delete entire components that touch the dragged rectangle."""
        if self.model.masks is None:
            return
        scene_start = self.canvas.mapToScene(start)
        scene_end = self.canvas.mapToScene(end)
        x0 = int(min(scene_start.x(), scene_end.x()))
        x1 = int(max(scene_start.x(), scene_end.x())) + 1
        y0 = int(min(scene_start.y(), scene_end.y()))
        y1 = int(max(scene_start.y(), scene_end.y())) + 1
        self.model.delete_components_touching_rect(self.model.index, x0, y0, x1, y1)
        self._update_view()

    # --------- additional utilities ---------
    def _on_close_window(self) -> None:
        """Close the window and quit the application."""
        self.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def choose_annotation_folder(self) -> str | None:
        """Prompt for a folder to load or save annotations."""
        path = QFileDialog.getExistingDirectory(self, "Select Annotation Folder")
        return path or None

    def _set_mask_folder(self) -> None:
        """Select and remember the default mask folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Mask Folder",
            config.get("mask_folder", ""),
        )
        if folder:
            config.set("mask_folder", folder)

    def close_current(self) -> None:
        """Close the currently loaded data and reset the view."""
        self.model = ZStackModel()
        self.slider.setEnabled(False)
        self.canvas.set_image(np.zeros((1, 1), dtype=np.uint8), reset_view=True)
        self.canvas.set_mask(None)
        self.statusBar().showMessage("Ready")

    def toggle_select_mode(self) -> None:
        """Toggle the brush selection mode."""
        self.brush_enabled = not self.brush_enabled

    def _delete_single_area(self, pos) -> None:
        """Delete the component under a clicked position."""
        if self.model.masks is None:
            return
        scene_pos = self.canvas.mapToScene(pos)
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        self.model.delete_components_touching_rect(self.model.index, x, y, x + 1, y + 1)
        self._update_view()

    def on_scroll(self, delta: int) -> None:
        """Scroll through slices using mouse wheel delta."""
        if delta > 0:
            self._prev_slice()
        else:
            self._next_slice()

    def update_masks(self) -> None:
        """Recompute component labels and refresh the display."""
        self.model.update_components()
        self._update_view()

    def _show_help(self) -> None:
        """Display a popup with available shortcuts and features."""
        text = (
            "Keyboard Shortcuts:\n"
            "  Arrow keys - previous/next slice\n"
            "  D/E - dilate/erode current mask\n"
            "  Z/X - undo/redo\n"
            "  P - toggle brush painting\n"
            "  [ and ] - change brush size\n"
            "  H - hand tool (panning)\n"
            "  Right click drag - delete masks touching drag rectangle\n\n"
            "Toolbar Buttons:\n"
            "  Prev/Next - move one slice backward or forward\n"
            "  Slider - jump to a specific slice index\n"
            "  Dilate/Erode - grow or shrink the mask; Strength sets iteration count\n"
            "  Strength - number of iterations for Dilate/Erode (1-10)\n"
            "  Filter < - remove components smaller than value in Filter spin\n"
            "  Filter spin - minimum pixel count for the small component filter\n"
            "  BG %/Bins + BG Filter - remove low intensity pixels using percentile and histogram bins\n"
            "  Stretch % + Stretch - histogram stretch (0 resets to original)\n"
            "  Blur + Blur value - apply Gaussian blur with given sigma\n"
            "  Show Original - toggle display of the unblurred image\n"
            "  Opacity Slider - mask overlay transparency\n"
            "  Clear Blur - restore the image without blur\n"
            "  Seed % + Seed - create mask seeds above intensity percentile\n"
            "  Diff %/Hist % + Int Grow - expand mask using intensity difference and optional histogram cutoff\n"
            "  Conn/Tol + Flood Grow - expand mask using flood fill with connectivity and tolerance\n\n"
            "Menus provide the same actions as the toolbar.\n"
            "Zoom with mouse wheel when over the image.\n"
            "Use Tools -> Script Editor to automate sequences of these actions"
        )
        QMessageBox.information(self, "Help", text)

    def _open_script_editor(self) -> None:
        """Open the script editor window as a non-modal dialog."""
        if getattr(self, "script_editor", None) is None:
            self.script_editor = ScriptEditor(self)
            self.script_editor.destroyed.connect(
                lambda: setattr(self, "script_editor", None)
            )
        self.script_editor.show()
        self.script_editor.raise_()
        self.script_editor.activateWindow()

