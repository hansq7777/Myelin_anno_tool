from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QSlider,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QLabel,
    QSpinBox,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QEvent, QPoint
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
from .file_helper import FileOpsMixin
from .morphology_helper import MorphologyMixin, IntGrowThread
from .script_helper import ScriptMixin


class MainController(QMainWindow, FileOpsMixin, MorphologyMixin, ScriptMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z-Stack Annotation (alpha)")
        self.model = ZStackModel()
        self.canvas = SliceCanvas()
        self.undo_stack: list[tuple[np.ndarray | None, float, str]] = []
        self.redo_stack: list[tuple[np.ndarray | None, float, str]] = []
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
        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(1, 10)
        self.strength_spin.setValue(1)
        self.skeleton_btn = QPushButton("Skeleton")
        self.skeleton_btn.clicked.connect(self._skeletonize_current)
        morph_layout.addWidget(self.dilate_btn)
        morph_layout.addWidget(self.erode_btn)
        morph_layout.addWidget(self.strength_spin)
        morph_layout.addWidget(self.skeleton_btn)
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
        self.show_orig_chk = QCheckBox("Show Original")
        self.show_orig_chk.toggled.connect(self._toggle_original)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._change_opacity)
        self.reverse_btn = QPushButton("Reverse")
        self.reverse_btn.clicked.connect(self._reverse_image)
        blur_layout.addWidget(self.show_orig_chk)
        blur_layout.addWidget(self.opacity_slider)
        blur_layout.addWidget(self.reverse_btn)
        ctrl.addLayout(blur_layout)

        res_layout = QVBoxLayout()
        self.resample_btn = QPushButton("Resample")
        self.resample_btn.clicked.connect(self._resample_stack)
        res_layout.addWidget(self.resample_btn)
        ctrl.addLayout(res_layout)

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
        grow_layout.addWidget(self.seed_thresh_edit)
        grow_layout.addWidget(self.seed_btn)
        grow_layout.addWidget(self.int_diff_edit)
        grow_layout.addWidget(self.int_hist_edit)
        grow_layout.addWidget(self.int_grow_btn)
        ctrl.addLayout(grow_layout)

        self.info_label = QLabel("")
        ctrl.addWidget(self.info_label)
        # Label to show current cursor position and pixel value
        self.cursor_label = QLabel("")
        ctrl.addWidget(self.cursor_label)
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
        clear_fg_act = mask_menu.addAction("Clear Foreground")
        clear_fg_act.triggered.connect(self._clear_foreground)
        clear_fg_act.setShortcuts(["Alt+D", "Meta+D"])

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
        reverse_act = image_menu.addAction("Reverse Intensities")
        reverse_act.triggered.connect(self._reverse_image)
        resample_act = image_menu.addAction("Resample…")
        resample_act.triggered.connect(self._resample_stack)

        linear_menu = self.menuBar().addMenu("Linear")
        frangi_act = linear_menu.addAction("Frangi Filter")
        frangi_act.triggered.connect(self._frangi_filter_prompt)
        sato_act = linear_menu.addAction("Sato Filter")
        sato_act.triggered.connect(self._sato_filter_prompt)
        meij_act = linear_menu.addAction("Meijering Filter")
        meij_act.triggered.connect(self._meijering_filter_prompt)
        thin_act = linear_menu.addAction("Thin Skeleton")
        thin_act.triggered.connect(
            lambda: self.script_skeletonize(algorithm="thin"))
        path_act = linear_menu.addAction("Shortest Path")
        path_act.triggered.connect(self._shortest_path_prompt)

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


    def closeEvent(self, event):
        if self._prompt_save_if_dirty():
            config.save()
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

    def _update_cursor_label(self, pos) -> None:
        """Update cursor position and pixel value label."""
        if self.model.data is None or self.model.original_data is None:
            self.cursor_label.setText("")
            return
        scene_pos = self.canvas.mapToScene(pos)
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        img = self.model.get_original_slice()
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            val = int(img[y, x])
            self.cursor_label.setText(f"Pos: ({x}, {y})  Value: {val}")
        else:
            self.cursor_label.setText("")

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

    def _push_undo(self, action: str = "", mask: np.ndarray | None = None) -> None:
        if mask is None and action != "stretch":
            if self.model.masks is None:
                return
            mask = self.model.masks.copy()
        self.undo_stack.append((mask, self.model.stretch_percent, action))
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
            if event.modifiers() in (Qt.AltModifier, Qt.MetaModifier):
                self._clear_foreground()
            else:
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


    def report_action(self, action: str, params: dict) -> None:
        path = os.path.basename(self.model.path or "")
        msg_params = ", ".join(f"{k}={v}" for k, v in params.items())
        print(
            f"{path} slice {self.model.index + 1}/{self.model.n_slices}: {action} {msg_params}"
        )

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        last_mask, last_stretch, action = self.undo_stack.pop()
        current_mask = self.model.masks.copy() if self.model.masks is not None else None
        self.redo_stack.append((current_mask, self.model.stretch_percent, action))
        if action == "stretch":
            if last_stretch <= 0:
                self.model.reset_contrast()
            else:
                self.model.histogram_stretch(last_stretch)
        else:
            if last_mask is not None:
                self.model.masks = last_mask
        if self.history:
            self.history.pop()
        self._update_view(reset_view=True)

    def _redo(self) -> None:
        if not self.redo_stack:
            return
        last_mask, last_stretch, action = self.redo_stack.pop()
        current_mask = self.model.masks.copy() if self.model.masks is not None else None
        self.undo_stack.append((current_mask, self.model.stretch_percent, action))
        if action == "stretch":
            if last_stretch <= 0:
                self.model.reset_contrast()
            else:
                self.model.histogram_stretch(last_stretch)
        else:
            if last_mask is not None:
                self.model.masks = last_mask
        self.history.append(action)
        self._update_view(reset_view=True)

    # --------- event filter ---------
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            self._handle_key(event)
            return True
        if obj in (self.canvas, self.canvas.viewport()) and event.type() == QEvent.MouseMove:
            self._update_cursor_label(event.pos())
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
            "  \u2318D or \u2325D - clear all foreground\n"
            "  P - toggle brush painting\n"
            "  [ and ] - change brush size\n"
            "  H - hand tool (panning)\n"
            "  Right click drag - delete masks touching drag rectangle\n\n"
            "Toolbar Buttons:\n"
            "  Prev/Next - move one slice backward or forward\n"
            "  Slider - jump to a specific slice index\n"
            "  Dilate/Erode - grow or shrink the mask; Strength sets iteration count\n"
            "  Skeleton - skeletonize the current mask\n"
            "  Threshold Abs/Norm - threshold by value or percentage\n"
            "  Strength - number of iterations for Dilate/Erode (1-10)\n"
            "  Filter < - remove components smaller than value in Filter spin\n"
            "  Filter spin - minimum pixel count for the small component filter\n"
            "  BG %/Bins + BG Filter - remove low intensity pixels using percentile and histogram bins\n"
            "  Stretch % + Stretch - histogram stretch (0 resets to original)\n"
            "  Show Original - toggle display of the unblurred image\n"
            "  Opacity Slider - mask overlay transparency\n"
            "  Reverse - invert pixel intensities\n"
            "  Seed % + Seed - create mask seeds above intensity percentile\n"
            "  Diff %/Hist % + Int Grow - expand mask using intensity difference and optional histogram cutoff\n"
            "  Conn/Tol + Flood Grow - expand mask using flood fill with connectivity and tolerance\n"
            "  Quick Save - save masks to the default path\n\n"
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

    def _frangi_filter_prompt(self) -> None:
        """Prompt for Frangi filter parameters and apply the filter."""
        start, ok = QInputDialog.getDouble(self, "Frangi Filter", "Sigma start:", 1.0, 0.1, 1e6, 2)
        if not ok:
            return
        end, ok = QInputDialog.getDouble(self, "Frangi Filter", "Sigma end:", 3.0, 0.1, 1e6, 2)
        if not ok:
            return
        step, ok = QInputDialog.getDouble(self, "Frangi Filter", "Sigma step:", 1.0, 0.1, 1e6, 2)
        if not ok:
            return
        thresh, ok = QInputDialog.getDouble(self, "Frangi Filter", "Threshold:", 0.5, 0.0, 1.0, 2)
        if not ok:
            return
        self.script_frangi_filter(start, end, step, thresh)

    def _sato_filter_prompt(self) -> None:
        """Prompt for Sato filter parameters and apply the filter."""
        start, ok = QInputDialog.getDouble(self, "Sato Filter", "Sigma start:", 1.0, 0.1, 1e6, 2)
        if not ok:
            return
        end, ok = QInputDialog.getDouble(self, "Sato Filter", "Sigma end:", 3.0, 0.1, 1e6, 2)
        if not ok:
            return
        step, ok = QInputDialog.getDouble(self, "Sato Filter", "Sigma step:", 1.0, 0.1, 1e6, 2)
        if not ok:
            return
        thresh, ok = QInputDialog.getDouble(self, "Sato Filter", "Threshold:", 0.5, 0.0, 1.0, 2)
        if not ok:
            return
        self.script_sato_filter(start, end, step, thresh)

    def _meijering_filter_prompt(self) -> None:
        """Prompt for Meijering filter parameters and apply the filter."""
        start, ok = QInputDialog.getDouble(self, "Meijering Filter", "Sigma start:", 1.0, 0.1, 1e6, 2)
        if not ok:
            return
        end, ok = QInputDialog.getDouble(self, "Meijering Filter", "Sigma end:", 3.0, 0.1, 1e6, 2)
        if not ok:
            return
        step, ok = QInputDialog.getDouble(self, "Meijering Filter", "Sigma step:", 1.0, 0.1, 1e6, 2)
        if not ok:
            return
        thresh, ok = QInputDialog.getDouble(self, "Meijering Filter", "Threshold:", 0.5, 0.0, 1.0, 2)
        if not ok:
            return
        self.script_meijering_filter(start, end, step, thresh)

    def _shortest_path_prompt(self) -> None:
        """Prompt for start and end coordinates for the shortest path."""
        y0, ok = QInputDialog.getInt(self, "Shortest Path", "Start Y:", 0, 0)
        if not ok:
            return
        x0, ok = QInputDialog.getInt(self, "Shortest Path", "Start X:", 0, 0)
        if not ok:
            return
        y1, ok = QInputDialog.getInt(self, "Shortest Path", "End Y:", 10, 0)
        if not ok:
            return
        x1, ok = QInputDialog.getInt(self, "Shortest Path", "End X:", 10, 0)
        if not ok:
            return
        self.script_shortest_path(y0, x0, y1, x1)

