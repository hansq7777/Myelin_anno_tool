from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QSlider,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QLabel,
    QSpinBox,
    QMessageBox,
    QInputDialog,
    QComboBox,
    QSizePolicy,
    QRubberBand,
)
from PyQt5.QtCore import Qt, QEvent, QPoint, QRect, QTimer
from PyQt5.QtGui import QCursor, QPixmap, QPainter, QPen, QColor
import threading
import sys
import os
import numpy as np
from ..models.zstack_model import ZStackModel
from ..views.canvas import SliceCanvas
from ..views.inline_volume_preview import InlineVolumePreview, VIEW_MODE_LABELS
from ..views.script_editor import ScriptEditor
from ..views.comparison_dialog import ComparisonDialog
from ..views.validation_dialog import ValidationDialog
from ..utils import morphology_tools
from ..utils.dialogs import question_with_shortcuts
from ..utils import config
from .file_helper import FileOpsMixin
from .morphology_helper import MorphologyMixin, IntGrowThread, FilterSmall3DThread
from .script_helper import ScriptMixin, QuickAutoThread
from .review_helper import ReviewMixin
from .volume_helper import VolumeMixin


class MainController(
    QMainWindow,
    FileOpsMixin,
    MorphologyMixin,
    ScriptMixin,
    ReviewMixin,
    VolumeMixin,
):
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
        self._painting_value: int | None = None
        self._last_pos = None
        self._temp_mask = None
        self._delete_hold = False
        self._delete_drag_start = None
        self.cancel_event = threading.Event()
        self.grow_thread: IntGrowThread | None = None
        self.filter_thread: FilterSmall3DThread | None = None
        self._filter_request_seq: int = 0
        self._active_filter_request_id: int | None = None
        self._pending_filter_undo_mask: np.ndarray | None = None
        self.script_editor: ScriptEditor | None = None
        self.quick_auto_cancel_event = threading.Event()
        self.quick_auto_thread: QuickAutoThread | None = None
        self._quick_auto_request_seq: int = 0
        self._active_quick_auto_request_id: int | None = None
        self._active_quick_auto_mode: str | None = None
        self._pending_quick_auto_undo_mask: np.ndarray | None = None
        self._quick_auto_snapshot_masks: np.ndarray | None = None
        self._quick_auto_snapshot_index: int = 0
        self._quick_auto_snapshot_label: str = ""
        self._pending_quick_auto_snapshot_masks: np.ndarray | None = None
        self._pending_quick_auto_snapshot_index: int = 0
        self._pending_quick_auto_snapshot_label: str = ""
        self._init_review_state()
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
        self.canvas.zoomAdjusted.connect(self._sync_inline_zoom_from_canvas)
        self.canvas.viewWindowChanged.connect(self._sync_inline_view_window_from_canvas)
        # Also filter events from the canvas viewport for painting
        self.canvas.viewport().installEventFilter(self)
        self.slider.installEventFilter(self)
        self.title_label.installEventFilter(self)
        self.menuBar().installEventFilter(self)
        self._delete_band = QRubberBand(QRubberBand.Rectangle, self.canvas.viewport())
        self._delete_band.setStyleSheet(
            "border: 2px solid rgba(255, 80, 80, 220);"
            "background-color: rgba(255, 80, 80, 45);"
        )
        self._delete_band.hide()
        self._inline_volume_enabled = False
        self.inline_volume_preview.zoomAdjusted.connect(self._sync_canvas_zoom_from_inline)
        self.inline_volume_preview.viewWindowChanged.connect(self._sync_canvas_view_window_from_inline)
        self.inline_volume_preview.rotationChanged.connect(self._sync_canvas_rotation_from_inline)
        self._inline_preview_timer = QTimer(self)
        self._inline_preview_timer.setSingleShot(True)
        self._inline_preview_timer.timeout.connect(self._refresh_inline_volume_preview)
        self._last_canvas_cursor_voxel: tuple[int, int] | None = None
        self._inline_preview_volume_signature: tuple | None = None

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
        self.inline_volume_chk = QCheckBox("3D Visualization")
        self.inline_volume_chk.toggled.connect(self._toggle_inline_volume_preview)
        self.inline_view_combo = QComboBox()
        for mode, label in VIEW_MODE_LABELS:
            self.inline_view_combo.addItem(label, mode)
        self.inline_view_combo.setCurrentIndex(1)
        self.inline_view_combo.currentIndexChanged.connect(self._on_inline_view_changed)
        self.inline_view_combo.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.slider)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(self.inline_volume_chk)
        nav_layout.addWidget(self.inline_view_combo)
        layout.addLayout(nav_layout)

        review_layout = QHBoxLayout()
        self.review_build_btn = QPushButton("Build Tracker")
        self.review_build_btn.clicked.connect(self._review_build_tracker_from_folders)
        self.review_open_btn = QPushButton("Open Review Tracker")
        self.review_open_btn.clicked.connect(self._open_review_tracker)
        self.review_prev_stack_btn = QPushButton("Prev Stack")
        self.review_prev_stack_btn.clicked.connect(self._review_prev_stack)
        self.review_next_stack_btn = QPushButton("Next Stack")
        self.review_next_stack_btn.clicked.connect(self._review_next_stack)
        self.review_next_unreviewed_btn = QPushButton("Next Unfinished")
        self.review_next_unreviewed_btn.clicked.connect(self._review_next_unfinished_stack)
        self.review_filter_combo = QComboBox()
        self.review_filter_combo.addItems(["All", "Unreviewed", "A", "B", "C"])
        self.review_filter_combo.currentTextChanged.connect(self._review_on_filter_changed)
        self.review_grade_combo = QComboBox()
        self.review_grade_combo.addItems(["Unreviewed", "A", "B", "C"])
        self.review_grade_combo.currentTextChanged.connect(self._review_on_grade_combo_changed)
        self.review_mark_a_btn = QPushButton("Mark A")
        self.review_mark_a_btn.clicked.connect(self._review_mark_a)
        self.review_mark_b_btn = QPushButton("Mark B")
        self.review_mark_b_btn.clicked.connect(self._review_mark_b)
        self.review_mark_c_btn = QPushButton("Mark C")
        self.review_mark_c_btn.clicked.connect(self._review_mark_c)
        self.review_save_corrected_btn = QPushButton("Save Corrected Mask")
        self.review_save_corrected_btn.clicked.connect(self._review_save_corrected_mask)
        self.review_export_final_btn = QPushButton("Export Final Masks")
        self.review_export_final_btn.clicked.connect(self._review_export_final_masks)
        self.quick_auto_preset_combo = QComboBox()
        self.quick_auto_preset_combo.addItem("Conservative", "conservative")
        self.quick_auto_preset_combo.addItem("Balanced", "balanced")
        self.quick_auto_preset_combo.addItem("Aggressive", "aggressive")
        self.quick_auto_preset_combo.setCurrentIndex(1)
        self.quick_auto_btn = QPushButton("Quick Auto Script")
        self.quick_auto_btn.clicked.connect(self._run_quick_auto_script)
        self.quick_auto_stack_btn = QPushButton("Quick Auto Stack")
        self.quick_auto_stack_btn.clicked.connect(self._run_quick_auto_stack)
        self.quick_auto_cancel_btn = QPushButton("Cancel Auto")
        self.quick_auto_cancel_btn.clicked.connect(self._cancel_quick_auto)
        self.quick_auto_cancel_btn.setEnabled(False)
        self.quick_auto_revert_btn = QPushButton("Revert Auto Snapshot")
        self.quick_auto_revert_btn.clicked.connect(self._restore_quick_auto_snapshot)
        self.quick_auto_revert_btn.setEnabled(False)
        self.review_info_label = QLabel("Review: tracker not loaded")
        review_layout.addWidget(self.review_build_btn)
        review_layout.addWidget(self.review_open_btn)
        review_layout.addWidget(self.review_prev_stack_btn)
        review_layout.addWidget(self.review_next_stack_btn)
        review_layout.addWidget(self.review_next_unreviewed_btn)
        review_layout.addWidget(QLabel("Filter"))
        review_layout.addWidget(self.review_filter_combo)
        review_layout.addWidget(QLabel("Grade"))
        review_layout.addWidget(self.review_grade_combo)
        review_layout.addWidget(self.review_mark_a_btn)
        review_layout.addWidget(self.review_mark_b_btn)
        review_layout.addWidget(self.review_mark_c_btn)
        review_layout.addWidget(self.review_save_corrected_btn)
        review_layout.addWidget(self.review_export_final_btn)
        review_layout.addWidget(QLabel("Auto Preset"))
        review_layout.addWidget(self.quick_auto_preset_combo)
        review_layout.addWidget(self.quick_auto_btn)
        review_layout.addWidget(self.quick_auto_stack_btn)
        review_layout.addWidget(self.quick_auto_cancel_btn)
        review_layout.addWidget(self.quick_auto_revert_btn)
        review_layout.addWidget(self.review_info_label, 1)
        layout.addLayout(review_layout)
        self._set_review_controls_enabled(False)

        self.inline_volume_preview = InlineVolumePreview()
        self.inline_volume_preview.hide()
        self.center_splitter = QSplitter(Qt.Horizontal)
        self.center_splitter.setChildrenCollapsible(False)
        self.center_splitter.addWidget(self.inline_volume_preview)
        self.center_splitter.addWidget(self.canvas)
        self.center_splitter.setStretchFactor(0, 0)
        self.center_splitter.setStretchFactor(1, 1)
        self.center_splitter.setSizes([0, 1000])
        layout.addWidget(self.center_splitter, 1)
        # -------- Controls --------
        ctrl = QHBoxLayout()

        # ---- Mask tools ----
        mask_layout = QVBoxLayout()
        self.dilate_btn = QPushButton("Dilate")
        self.dilate_btn.clicked.connect(self._dilate_current)
        self.erode_btn = QPushButton("Erode")
        self.erode_btn.clicked.connect(self._erode_current)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self._close_current)
        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(1, 10)
        self.strength_spin.setValue(1)
        self.skeleton_btn = QPushButton("Skeleton")
        self.skeleton_btn.clicked.connect(self._skeletonize_current)
        self.filter_btn = QPushButton("Filter <")
        self.filter_btn.clicked.connect(self._filter_small)
        self.filter_spin = QSpinBox()
        self.filter_spin.setRange(1, 10000)
        self.filter_spin.setValue(100)
        self.clear_prev_btn = QPushButton("Clear <= Slice")
        self.clear_prev_btn.clicked.connect(self._clear_to_current_slice)
        self.clear_next_btn = QPushButton("Clear >= Slice")
        self.clear_next_btn.clicked.connect(self._clear_from_current_slice)
        mask_layout.addWidget(self.dilate_btn)
        mask_layout.addWidget(self.erode_btn)
        mask_layout.addWidget(self.close_btn)
        mask_layout.addWidget(self.strength_spin)
        mask_layout.addWidget(self.skeleton_btn)
        mask_layout.addWidget(self.filter_btn)
        mask_layout.addWidget(self.filter_spin)
        mask_layout.addWidget(self.clear_prev_btn)
        mask_layout.addWidget(self.clear_next_btn)
        self.mask_vis_label = QLabel("Mask Visibility")
        self.mask_vis_slider = QSlider(Qt.Horizontal)
        self.mask_vis_slider.setRange(0, 100)
        self.mask_vis_slider.setValue(50)
        self.mask_vis_slider.valueChanged.connect(self._change_mask_visibility)
        mask_layout.addWidget(self.mask_vis_label)
        mask_layout.addWidget(self.mask_vis_slider)
        ctrl.addLayout(mask_layout)

        # ---- Thresholding ----
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

        # ---- Region growing ----
        grow_layout = QVBoxLayout()
        self.seed_thresh_edit = QLineEdit()
        self.seed_thresh_edit.setPlaceholderText("Seed %")
        self.seed_pix_edit = QLineEdit()
        self.seed_pix_edit.setPlaceholderText("Pix %")
        self.seed_btn = QPushButton("Seed")
        self.seed_btn.clicked.connect(self._seed_current)
        self.int_diff_edit = QLineEdit()
        self.int_diff_edit.setPlaceholderText("Diff %")
        self.int_hist_edit = QLineEdit()
        self.int_hist_edit.setPlaceholderText("Hist %")
        self.int_grow_btn = QPushButton("Int Grow")
        self.int_grow_btn.clicked.connect(self._grow_intensity)
        grow_layout.addWidget(self.seed_thresh_edit)
        grow_layout.addWidget(self.seed_pix_edit)
        grow_layout.addWidget(self.seed_btn)
        grow_layout.addWidget(self.int_diff_edit)
        grow_layout.addWidget(self.int_hist_edit)
        grow_layout.addWidget(self.int_grow_btn)
        ctrl.addLayout(grow_layout)

        # ---- Image adjustments ----
        img_layout = QVBoxLayout()
        self.bg_percentile_edit = QLineEdit()
        self.bg_percentile_edit.setPlaceholderText("BG %")
        self.bg_bins_edit = QLineEdit()
        self.bg_bins_edit.setPlaceholderText("Bins")
        self.bg_filter_button = QPushButton("BG Filter")
        self.bg_filter_button.clicked.connect(self._apply_bg_filter)
        self.stretch_edit = QLineEdit()
        self.stretch_edit.setPlaceholderText("Stretch %")
        self.stretch_button = QPushButton("Stretch")
        self.stretch_button.clicked.connect(self._apply_stretch)
        self.reverse_btn = QPushButton("Reverse")
        self.reverse_btn.clicked.connect(self._reverse_image)
        self.resample_btn = QPushButton("Resample")
        self.resample_btn.clicked.connect(self._resample_stack)
        img_layout.addWidget(self.bg_percentile_edit)
        img_layout.addWidget(self.bg_bins_edit)
        img_layout.addWidget(self.bg_filter_button)
        img_layout.addWidget(self.stretch_edit)
        img_layout.addWidget(self.stretch_button)
        img_layout.addWidget(self.reverse_btn)
        img_layout.addWidget(self.resample_btn)
        ctrl.addLayout(img_layout)

        # ---- Display ----
        disp_layout = QVBoxLayout()
        self.show_orig_chk = QCheckBox("Show Original")
        self.show_orig_chk.toggled.connect(self._toggle_original)
        disp_layout.addWidget(self.show_orig_chk)
        ctrl.addLayout(disp_layout)

        self.info_label = QLabel("")
        ctrl.addWidget(self.info_label)
        # Label to show current cursor position and pixel value
        self.cursor_label = QLabel("")
        cursor_width = self.cursor_label.fontMetrics().horizontalAdvance(
            "Pos: (00000, 00000)  Value: 65535"
        )
        self.cursor_label.setFixedWidth(cursor_width + 16)
        self.cursor_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.cursor_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        ctrl.addWidget(self.cursor_label)
        layout.addLayout(ctrl)
        self.setCentralWidget(central)

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("File")
        open_act = file_menu.addAction("Open…")
        open_act.triggered.connect(self._open_file)
        open_act.setShortcuts(["Ctrl+O", "Meta+O"])

        import_czi_act = file_menu.addAction("Import CZI…")
        import_czi_act.triggered.connect(self._import_czi_file)

        meta_act = file_menu.addAction("Export CZI Metadata…")
        meta_act.triggered.connect(self._export_czi_metadata)

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

        info_act = file_menu.addAction("Stack Info")
        info_act.triggered.connect(self._show_stack_info)

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
        close_act = mask_menu.addAction("Close")
        close_act.triggered.connect(self._close_current)
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
        raw_menu = linear_menu.addMenu("Raw Image")
        seed_menu = linear_menu.addMenu("Seed Mask")

        frangi_act = raw_menu.addAction("Frangi Filter")
        frangi_act.triggered.connect(self._frangi_filter_prompt)
        sato_act = raw_menu.addAction("Sato Filter")
        sato_act.triggered.connect(self._sato_filter_prompt)
        meij_act = raw_menu.addAction("Meijering Filter")
        meij_act.triggered.connect(self._meijering_filter_prompt)

        thin_act = linear_menu.addAction("Thin Skeleton")
        thin_act.triggered.connect(
            lambda: self.script_skeletonize(algorithm="thin"))

        felz_act = raw_menu.addAction("Felzenszwalb")
        felz_act.triggered.connect(
            lambda: self.script_felzenszwalb())
        water_act = seed_menu.addAction("Watershed IFT")
        water_act.triggered.connect(
            lambda: self.script_watershed_ift())
        fmm_act = seed_menu.addAction("scikit-fmm")
        fmm_act.triggered.connect(
            lambda: self.script_scikit_fmm())
        march_act = seed_menu.addAction("Fast Marching")
        march_act.triggered.connect(
            lambda: self.script_fast_marching())
        ridge_act = raw_menu.addAction("OpenCV Ridge")
        ridge_act.triggered.connect(lambda: self.script_opencv_ridge())
        steger_act = raw_menu.addAction("Steger Ridge")
        steger_act.triggered.connect(lambda: self.script_steger_ridge())
        chan_act = raw_menu.addAction("Chan-Vese")
        chan_act.triggered.connect(lambda: self.script_chan_vese())
        ced_act = raw_menu.addAction("CED Filter")
        ced_act.triggered.connect(lambda: self.script_ced_filter())
        tube_act = raw_menu.addAction("TubeTK Tubes")
        tube_act.triggered.connect(lambda: self.script_tubetk_segment())
        seed_act = seed_menu.addAction("TubeTK Seed Path")
        seed_act.triggered.connect(lambda: self.script_tubetk_seed_path())
        hess_act = raw_menu.addAction("Hessian Filter")
        hess_act.triggered.connect(lambda: self.script_hessian_filter())
        gabor_act = raw_menu.addAction("Gabor Filter")
        gabor_act.triggered.connect(lambda: self.script_gabor_filter())
        cvgab_act = raw_menu.addAction("OpenCV Gabor")
        cvgab_act.triggered.connect(lambda: self.script_cv_gabor_filter())
        st_act = raw_menu.addAction("Structure Tensor")
        st_act.triggered.connect(lambda: self.script_structure_tensor())

        tool_menu = self.menuBar().addMenu("Tools")
        script_act = tool_menu.addAction("Script Editor")
        script_act.triggered.connect(self._open_script_editor)
        # Support Command+E and Option+E on macOS, and Alt+E elsewhere.
        # Ctrl+E is mapped to Command+E automatically on macOS, so include
        # it alongside Alt/Meta for cross-platform compatibility.
        script_act.setShortcuts(["Alt+E", "Ctrl+E", "Meta+E"])
        self.quick_auto_act = tool_menu.addAction("Run Quick Auto Script")
        self.quick_auto_act.triggered.connect(self._run_quick_auto_script)
        self.quick_auto_act.setShortcuts(["Alt+Q"])
        self.quick_auto_stack_act = tool_menu.addAction("Run Quick Auto On Stack…")
        self.quick_auto_stack_act.triggered.connect(self._run_quick_auto_stack)
        self.quick_auto_stack_act.setShortcuts(["Alt+W"])
        self.quick_auto_cancel_act = tool_menu.addAction("Cancel Quick Auto")
        self.quick_auto_cancel_act.triggered.connect(self._cancel_quick_auto)
        self.quick_auto_cancel_act.setShortcuts(["Alt+Shift+W"])
        self.quick_auto_cancel_act.setEnabled(False)
        self.quick_auto_revert_act = tool_menu.addAction("Revert Quick Auto Snapshot")
        self.quick_auto_revert_act.triggered.connect(self._restore_quick_auto_snapshot)
        self.quick_auto_revert_act.setShortcuts(["Alt+Shift+Q"])
        self.quick_auto_revert_act.setEnabled(False)
        compare_act = tool_menu.addAction("Strategy Comparison")
        compare_act.triggered.connect(self._open_comparison_dialog)
        # Support Command+R and Option+R on macOS, and Alt+R elsewhere.
        # Ctrl+R is mapped to Command+R automatically on macOS, so include
        # it alongside Alt/Meta for cross-platform compatibility.
        compare_act.setShortcuts(["Alt+R", "Ctrl+R", "Meta+R"])

        validate_act = tool_menu.addAction("Validation Viewer")
        validate_act.triggered.connect(self._open_validation_dialog)
        # Support Command+T and Option+T on macOS, and Alt+T elsewhere.
        # Ctrl+T is mapped to Command+T automatically on macOS, so include
        # it alongside Alt/Meta for cross-platform compatibility.
        validate_act.setShortcuts(["Alt+T", "Ctrl+T", "Meta+T"])
        current_3d_act = tool_menu.addAction("3D Inspector (Current Stack)")
        current_3d_act.triggered.connect(self._open_volume_inspector_current)
        current_3d_act.setShortcuts(["Alt+G", "Ctrl+G", "Meta+G"])
        matching_3d_act = tool_menu.addAction("3D Inspector (Matching Inference)")
        matching_3d_act.triggered.connect(self._open_volume_inspector_matching)
        matching_3d_act.setShortcuts(["Alt+Shift+G", "Ctrl+Shift+G", "Meta+Shift+G"])

        review_menu = self.menuBar().addMenu("Review")
        review_build_act = review_menu.addAction("Build Tracker from Folders…")
        review_build_act.triggered.connect(self._review_build_tracker_from_folders)
        review_open_act = review_menu.addAction("Open Tracker…")
        review_open_act.triggered.connect(self._open_review_tracker)
        review_prev_act = review_menu.addAction("Previous Stack")
        review_prev_act.triggered.connect(self._review_prev_stack)
        review_prev_act.setShortcuts(["Alt+,"])
        review_next_act = review_menu.addAction("Next Stack")
        review_next_act.triggered.connect(self._review_next_stack)
        review_next_act.setShortcuts(["Alt+."])
        review_next_unreviewed_act = review_menu.addAction("Next Unfinished Stack")
        review_next_unreviewed_act.triggered.connect(self._review_next_unfinished_stack)
        review_next_unreviewed_act.setShortcuts(["Alt+/"])
        mark_a_act = review_menu.addAction("Mark A")
        mark_a_act.triggered.connect(self._review_mark_a)
        mark_a_act.setShortcuts(["Alt+1"])
        mark_b_act = review_menu.addAction("Mark B")
        mark_b_act.triggered.connect(self._review_mark_b)
        mark_b_act.setShortcuts(["Alt+2"])
        mark_c_act = review_menu.addAction("Mark C")
        mark_c_act.triggered.connect(self._review_mark_c)
        mark_c_act.setShortcuts(["Alt+3"])
        save_corr_act = review_menu.addAction("Save Corrected Mask")
        save_corr_act.triggered.connect(self._review_save_corrected_mask)
        save_corr_act.setShortcuts(["Alt+Shift+S"])
        export_final_act = review_menu.addAction("Export Final Masks")
        export_final_act.triggered.connect(self._review_export_final_masks)
        export_final_act.setShortcuts(["Alt+Shift+F"])

        help_menu = self.menuBar().addMenu("Help")
        help_act = help_menu.addAction("Shortcuts && Features")
        help_act.triggered.connect(self._show_help)


    def closeEvent(self, event):
        if self._prompt_save_if_dirty():
            self._close_volume_windows()
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
        mask = None
        if self.model.masks is not None:
            try:
                mask = self.model.get_mask()
            except Exception:
                # Keep UI responsive even if an external mask file has
                # unexpected depth/shape; image display should remain usable.
                mask = None
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
        if hasattr(self, "_review_sync_view_badges"):
            self._review_sync_view_badges()
        self._schedule_inline_volume_preview_refresh()
        self._refresh_inline_volume_locator()

    def _update_cursor_label(self, pos) -> None:
        """Update cursor position and pixel value label."""
        if self.model.data is None or self.model.original_data is None:
            self.cursor_label.setText("")
            self._last_canvas_cursor_voxel = None
            self._refresh_inline_volume_locator()
            return
        scene_pos = self.canvas.mapToScene(pos)
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        img = self.model.get_original_slice()
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            val = int(img[y, x])
            self.cursor_label.setText(f"Pos: ({x}, {y})  Value: {val}")
            self._last_canvas_cursor_voxel = (x, y)
        else:
            self.cursor_label.setText("")
            self._last_canvas_cursor_voxel = None
        self._refresh_inline_volume_locator()

    def _toggle_inline_volume_preview(self, enabled: bool) -> None:
        self._inline_volume_enabled = bool(enabled)
        self.inline_volume_preview.setVisible(self._inline_volume_enabled)
        self.inline_view_combo.setEnabled(self._inline_volume_enabled)
        if not self._inline_volume_enabled:
            self._inline_preview_timer.stop()
            self.inline_volume_preview.clear_preview()
            self._inline_preview_volume_signature = None
            self.canvas.set_view_rotation(0.0, emit_signal=False)
            self.center_splitter.setSizes([0, max(1, self.center_splitter.width())])
            return
        total = max(sum(self.center_splitter.sizes()), self.center_splitter.width(), 960)
        left = max(300, total // 3)
        self.center_splitter.setSizes([left, max(420, total - left)])
        self._schedule_inline_volume_preview_refresh(delay_ms=10)

    def _on_inline_view_changed(self, _index: int) -> None:
        if not getattr(self, "_inline_volume_enabled", False):
            return
        if (self.inline_view_combo.currentData() or "xy") != "xy":
            self.canvas.set_view_rotation(0.0, emit_signal=False)
        self._schedule_inline_volume_preview_refresh(delay_ms=10)

    def _schedule_inline_volume_preview_refresh(self, *, delay_ms: int = 80) -> None:
        if not getattr(self, "_inline_volume_enabled", False):
            return
        self._inline_preview_timer.start(max(0, int(delay_ms)))

    def _current_spacing_xyz(self) -> tuple[float, float, float]:
        return self.model.get_pixel_sizes() or (1.0, 1.0, 1.0)

    def _sync_inline_zoom_from_canvas(self, factor: float) -> None:
        if not getattr(self, "_inline_volume_enabled", False):
            return
        if (self.inline_view_combo.currentData() or "xy") == "xy":
            return
        self.inline_volume_preview.apply_zoom_factor(float(factor), emit_signal=False)

    def _sync_canvas_zoom_from_inline(self, factor: float) -> None:
        if (self.inline_view_combo.currentData() or "xy") == "xy":
            return
        self.canvas.apply_zoom_factor(float(factor), emit_signal=False)

    def _sync_canvas_view_window_from_inline(
        self,
        center_x_norm: float,
        center_y_norm: float,
        frac_x: float,
        frac_y: float,
    ) -> None:
        if not getattr(self, "_inline_volume_enabled", False):
            return
        if (self.inline_view_combo.currentData() or "xy") != "xy":
            return
        self.canvas.set_normalized_view_window(
            center_xy_norm=(float(center_x_norm), float(center_y_norm)),
            visible_fraction_xy=(float(frac_x), float(frac_y)),
            emit_signal=False,
        )

    def _sync_canvas_rotation_from_inline(self, yaw_deg: float) -> None:
        if not getattr(self, "_inline_volume_enabled", False):
            return
        if (self.inline_view_combo.currentData() or "xy") != "xy":
            self.canvas.set_view_rotation(0.0, emit_signal=False)
            return
        self.canvas.set_view_rotation(float(yaw_deg), emit_signal=False)

    def _sync_inline_view_window_from_canvas(
        self,
        center_x_norm: float,
        center_y_norm: float,
        frac_x: float,
        frac_y: float,
    ) -> None:
        if not getattr(self, "_inline_volume_enabled", False):
            return
        if (self.inline_view_combo.currentData() or "xy") != "xy":
            return
        self.inline_volume_preview.set_planar_view_window(
            center_xy_norm=(float(center_x_norm), float(center_y_norm)),
            visible_fraction_xy=(float(frac_x), float(frac_y)),
        )

    def _refresh_inline_volume_preview(self) -> None:
        if not getattr(self, "_inline_volume_enabled", False):
            return
        raw = self.model.original_data if self.model.original_data is not None else self.model.data
        if raw is None:
            self.inline_volume_preview.clear_preview()
            self._inline_preview_volume_signature = None
            return
        view_mode = self.inline_view_combo.currentData() or "xy"
        spacing = self._current_spacing_xyz()
        volume_signature = (
            int(self.model.image_revision),
            int(self.model.mask_revision),
            int(id(raw)),
            int(id(self.model.masks)),
            tuple(int(v) for v in raw.shape),
            None if self.model.masks is None else tuple(int(v) for v in self.model.masks.shape),
            tuple(round(float(v), 6) for v in spacing),
        )
        if volume_signature != self._inline_preview_volume_signature:
            self.inline_volume_preview.set_view_mode(view_mode, render_if_ready=False)
            self.inline_volume_preview.set_volume(
                raw,
                self.model.masks,
                spacing,
                cache_key=volume_signature,
            )
            self._inline_preview_volume_signature = volume_signature
        else:
            self.inline_volume_preview.set_view_mode(view_mode)
        center_xy, visible_xy = self.canvas.normalized_view_window()
        self._sync_inline_view_window_from_canvas(
            center_xy[0],
            center_xy[1],
            visible_xy[0],
            visible_xy[1],
        )
        self._refresh_inline_volume_locator()

    def _refresh_inline_volume_locator(self) -> None:
        if not getattr(self, "_inline_volume_enabled", False):
            return
        if self.model.data is None:
            self.inline_volume_preview.clear_preview()
            return
        spacing = self._current_spacing_xyz()
        self.inline_volume_preview.set_current_slice(self.model.index, spacing)
        if self._last_canvas_cursor_voxel is None:
            self.inline_volume_preview.clear_locator()
            return
        x, y = self._last_canvas_cursor_voxel
        img = self.model.get_original_slice()
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            self.inline_volume_preview.set_locator(x, y, self.model.index, spacing)
        else:
            self.inline_volume_preview.clear_locator()

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
    def _handle_key_press(self, event):
        if not self.slider.isEnabled():
            return
        if event.modifiers() == Qt.AltModifier and event.key() == Qt.Key_Period:
            self._review_next_stack()
            return
        if event.modifiers() == Qt.AltModifier and event.key() == Qt.Key_Comma:
            self._review_prev_stack()
            return
        if event.modifiers() == Qt.AltModifier and event.key() == Qt.Key_Slash:
            self._review_next_unfinished_stack()
            return
        if event.modifiers() == Qt.AltModifier and event.key() == Qt.Key_1:
            self._review_mark_a()
            return
        if event.modifiers() == Qt.AltModifier and event.key() == Qt.Key_2:
            self._review_mark_b()
            return
        if event.modifiers() == Qt.AltModifier and event.key() == Qt.Key_3:
            self._review_mark_c()
            return
        if event.modifiers() == Qt.AltModifier and event.key() == Qt.Key_Q:
            self._run_quick_auto_script()
            return
        if event.modifiers() == Qt.AltModifier and event.key() == Qt.Key_W:
            self._run_quick_auto_stack()
            return
        if event.modifiers() == (Qt.AltModifier | Qt.ShiftModifier) and event.key() == Qt.Key_Q:
            self._restore_quick_auto_snapshot()
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
            elif not event.isAutoRepeat():
                if self._painting:
                    self._end_paint()
                self._delete_hold = True
                self._sync_tool_cursor()
                self.statusBar().showMessage("Delete component(s): click or drag")
        elif event.key() == Qt.Key_E:
            self._erode_current()
        elif event.key() == Qt.Key_Z:
            self._undo()
        elif event.key() == Qt.Key_X:
            self._redo()
        elif event.key() == Qt.Key_P:
            if event.isAutoRepeat():
                return
            if self._painting:
                self._end_paint()
            self.brush_enabled = not self.brush_enabled
            self._sync_tool_cursor()
            self.statusBar().showMessage(
                "Brush ON: left add, right erase" if self.brush_enabled else "Brush OFF"
            )
        elif event.key() == Qt.Key_H:
            if event.isAutoRepeat():
                return
            if self._painting:
                self._end_paint()
            self.brush_enabled = False
            self.canvas.setDragMode(self.canvas.ScrollHandDrag)
            self._sync_tool_cursor()
            self.statusBar().showMessage("Hand tool")
        elif event.key() == Qt.Key_BracketLeft:
            self.brush_size = max(1, self.brush_size - 1)
            self._sync_tool_cursor()
        elif event.key() == Qt.Key_BracketRight:
            self.brush_size += 1
            self._sync_tool_cursor()

    def _handle_key_release(self, event) -> None:
        if not self.slider.isEnabled():
            return
        if event.key() == Qt.Key_D and self._delete_hold and not event.isAutoRepeat():
            self._delete_hold = False
            if self._delete_drag_start is None:
                self._sync_tool_cursor()
                if self.brush_enabled:
                    self.statusBar().showMessage("Brush ON: left add, right erase")
                else:
                    self.statusBar().showMessage("Ready")


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
                self.model.replace_masks(last_mask, dirty=True)
        if self.history:
            self.history.pop()
        self._update_view()

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
                self.model.replace_masks(last_mask, dirty=True)
        self.history.append(action)
        self._update_view()

    def _toggle_window_fit_screen(self) -> None:
        """Toggle between normal and maximized window state."""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # --------- event filter ---------
    def eventFilter(self, obj, event):
        if (
            obj in (self.title_label, self.menuBar())
            and event.type() == QEvent.MouseButtonDblClick
            and event.button() == Qt.LeftButton
        ):
            self._toggle_window_fit_screen()
            return True
        if event.type() == QEvent.KeyPress:
            self._handle_key_press(event)
            return True
        if event.type() == QEvent.KeyRelease:
            self._handle_key_release(event)
            return True
        if obj in (self.canvas, self.canvas.viewport()) and event.type() == QEvent.MouseMove:
            self._update_cursor_label(event.pos())
        if (
            obj in (self.canvas, self.canvas.viewport())
            and self._delete_tool_active()
            and event.type() in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease)
        ):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._start_delete_drag(event.pos())
                return True
            if (
                event.type() == QEvent.MouseMove
                and self._delete_drag_start is not None
                and event.buttons() & Qt.LeftButton
            ):
                self._update_delete_drag(event.pos())
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._finish_delete_drag(event.pos())
                return True
        if (
            self.brush_enabled
            and not self._delete_tool_active()
            and obj in (self.canvas, self.canvas.viewport())
            and event.type() in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease)
        ):
            if event.type() == QEvent.MouseButtonPress and event.button() in (Qt.LeftButton, Qt.RightButton):
                self._start_paint(event.pos(), 1 if event.button() == Qt.LeftButton else 0)
                return True
            if (
                event.type() == QEvent.MouseMove
                and self._painting
                and event.buttons() & (Qt.LeftButton | Qt.RightButton)
            ):
                self._continue_paint(event.pos())
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() in (Qt.LeftButton, Qt.RightButton):
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
        yy, xx = np.ogrid[y0:y1, x0:x1]
        radius = max(1.0, self.brush_size / 2.0)
        circle = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
        value = 0 if self._painting_value == 0 else 1
        patch = mask[y0:y1, x0:x1]
        patch[circle] = value
        mask[y0:y1, x0:x1] = patch
        self.canvas.set_mask(mask)

    def _start_paint(self, pos, value: int) -> None:
        if self._painting:
            return
        self._push_undo("paint")
        self._painting = True
        self._painting_value = 1 if value > 0 else 0
        self._last_pos = pos
        self._temp_mask = self.model.get_mask().copy()
        self._sync_tool_cursor()
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
        self._painting_value = None
        self._update_view()
        self._sync_tool_cursor()

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
        self._push_undo("delete_rect")
        changed = self.model.delete_components_touching_rect(self.model.index, x0, y0, x1, y1)
        if not changed:
            self._discard_last_undo()
            return
        self._update_view()

    def _discard_last_undo(self) -> None:
        if self.undo_stack:
            self.undo_stack.pop()
        if self.history:
            self.history.pop()

    def _delete_tool_active(self) -> bool:
        return self._delete_hold or self._delete_drag_start is not None

    def _start_delete_drag(self, pos) -> None:
        if self.model.masks is None:
            return
        self._delete_drag_start = QPoint(pos)
        self._delete_band.setGeometry(QRect(self._delete_drag_start, self._delete_drag_start))
        self._delete_band.show()
        self._sync_tool_cursor()

    def _update_delete_drag(self, pos) -> None:
        if self._delete_drag_start is None:
            return
        self._delete_band.setGeometry(QRect(self._delete_drag_start, pos).normalized())

    def _finish_delete_drag(self, pos) -> None:
        if self._delete_drag_start is None:
            return
        start = QPoint(self._delete_drag_start)
        self._update_delete_drag(pos)
        self._delete_band.hide()
        self._delete_drag_start = None
        self._delete_rect(start, pos)
        self._sync_tool_cursor()

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
        self._sync_tool_cursor()

    def _delete_single_area(self, pos) -> None:
        """Delete the component under a clicked position."""
        self._delete_rect(pos, pos)

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
            "  Alt+, / Alt+. - previous/next review stack\n"
            "  Alt+/ - jump to the next unfinished zstack in the fixed queue\n"
            "  Alt+1 / Alt+2 / Alt+3 - mark review A/B/C\n"
            "  Alt+Shift+F - export reviewed final masks (A/B/C)\n"
            "  Alt+Q - run quick auto script (seed->dilate->bg->grow->bg)\n"
            "  Alt+W - run quick auto on current stack/range\n"
            "  Alt+Shift+Q - revert to pre-auto snapshot\n"
            "  Alt+G - open 3D inspector for the current stack\n"
            "  Alt+Shift+G - open 3D inspector with matching inference assets\n"
            "  G (when 3D preview focused) - reset left 3D view to the selected base orientation\n"
            "  Left 3D preview: left drag pan, right drag rotate, wheel zoom, Ctrl+left = rotate on macOS\n"
            "  D (hold) - click/drag delete connected component(s)\n"
            "  E - erode current mask\n"
            "  Z/X - undo/redo\n"
            "  \u2318D or \u2325D - clear foreground on current slice\n"
            "  P - toggle brush painting (left add / right erase)\n"
            "  [ and ] - change brush size\n"
            "  H - hand tool (panning)\n"
            "  Hold D + left drag - box delete with preview\n\n"
            "Toolbar Buttons:\n"
            "  Prev/Next - move one slice backward or forward\n"
            "  Prev Stack/Next Stack - move between raw+prediction pairs from the review tracker\n"
            "  Next Unfinished - skip completed stacks and jump to the next pending zstack\n"
            "  Build Tracker - create/refresh tracker from raw+prediction folders\n"
            "  Review queue is fixed when the tracker is built; low-foreground predictions are weighted earlier\n"
            "  Export Final Masks - build unified final mask set from reviewed items\n"
            "  Auto Preset - choose conservative/balanced/aggressive parameters\n"
            "  Quick Auto Script - run default cleanup pipeline on current slice\n"
            "  Quick Auto Stack - run quick auto on all slices or a selected range\n"
            "  Cancel Auto - cancel the current background quick auto run\n"
            "  Revert Auto Snapshot - restore mask state before the last auto run\n"
            "  Slider - jump to a specific slice index\n"
            "  Dilate/Erode/Skeleton - basic mask operations; Strength sets iteration count\n"
            "  Filter </spin - remove small connected components by total 3D voxel count\n"
            "  Clear <= Slice - clear labels on current and all previous slices\n"
            "  Clear >= Slice - clear labels on current and all following slices\n"
            "  Threshold Abs/Norm - threshold by value or percentage\n"
            "  Seed %/Pix % + Seed - create mask seeds above intensity percentile\n"
            "  Diff %/Hist % + Int Grow - expand mask using intensity difference and optional histogram cutoff\n"
            "  BG %/Bins + BG Filter - remove low intensity pixels using percentile and histogram bins\n"
            "  Stretch % + Stretch - histogram stretch (0 resets to original)\n"
            "  Reverse/Resample - adjust intensity or z-spacing\n"
            "  Show Original + Opacity Slider - control display transparency\n"
            "  Quick Save - save masks to the default path\n\n"
            "Menus provide the same actions as the toolbar.\n"
            "Zoom with mouse wheel when over the image.\n"
            "Use Tools -> Script Editor to automate sequences of these actions.\n"
            "Use Tools -> 3D Inspector to render raw stacks and prediction masks as a 3D view.\n"
            "Use Review controls to classify stacks as A/B/C and save corrected masks.\n"
            "Saving corrected masks marks the item as completed."
        )
        QMessageBox.information(self, "Help", text)

    def _run_quick_auto_script(self) -> None:
        if self.model.data is None:
            QMessageBox.warning(self, "Quick Auto Script", "Please load an image first.")
            return
        if self._quick_auto_is_running():
            self.statusBar().showMessage("Quick auto is already running in background.")
            return
        self._start_quick_auto_job(
            mode="single",
            indices=[self.model.index],
            label="single-slice quick auto",
            params=self._quick_auto_params_for_selected_preset(),
        )

    def _sync_tool_cursor(self) -> None:
        if self._delete_tool_active():
            self.canvas.setDragMode(self.canvas.NoDrag)
            self.canvas.viewport().setCursor(self._make_delete_cursor())
            return
        if not self.brush_enabled:
            self.canvas.viewport().unsetCursor()
            self.canvas.setDragMode(self.canvas.ScrollHandDrag)
            return
        self.canvas.setDragMode(self.canvas.NoDrag)
        self.canvas.viewport().setCursor(self._make_brush_cursor())

    def _make_brush_cursor(self) -> QCursor:
        radius = max(3, int(round(self.brush_size / 2.0)))
        size = max(18, radius * 2 + 8)
        center = size // 2
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        color = QColor(255, 80, 80, 220) if self._painting_value == 0 else QColor(0, 220, 120, 220)
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
        painter.setPen(QPen(color, 1))
        painter.drawLine(center - 3, center, center + 3, center)
        painter.drawLine(center, center - 3, center, center + 3)
        painter.end()
        return QCursor(pix, center, center)

    def _make_delete_cursor(self) -> QCursor:
        size = 24
        center = size // 2
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        color = QColor(255, 80, 80, 230)
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(4, 4, size - 8, size - 8)
        painter.drawLine(center, 4, center, size - 4)
        painter.drawLine(4, center, size - 4, center)
        painter.end()
        return QCursor(pix, center, center)

    def _run_quick_auto_stack(self) -> None:
        if self.model.data is None:
            QMessageBox.warning(self, "Quick Auto Stack", "Please load an image first.")
            return
        if self._quick_auto_is_running():
            self.statusBar().showMessage("Quick auto is already running in background.")
            return

        total = self.model.n_slices
        mode, ok = QInputDialog.getItem(
            self,
            "Quick Auto Stack",
            "Select run mode:",
            ["All slices", "Slice range...", "Key slices (first/middle/last)"],
            0,
            False,
        )
        if not ok:
            return

        if mode == "All slices":
            indices = list(range(total))
        elif mode == "Slice range...":
            start, ok = QInputDialog.getInt(
                self, "Start Slice", f"Start slice (1-{total}):", 1, 1, total
            )
            if not ok:
                return
            end, ok = QInputDialog.getInt(
                self, "End Slice", f"End slice ({start}-{total}):", total, start, total
            )
            if not ok:
                return
            indices = list(range(start - 1, end))
        else:
            indices = sorted(set([0, total // 2, total - 1]))

        if not indices:
            return
        self._start_quick_auto_job(
            mode="stack",
            indices=indices,
            label="stack quick auto",
            params=self._quick_auto_params_for_selected_preset(),
        )

    def _quick_auto_params_for_selected_preset(self) -> dict[str, object]:
        preset = self.quick_auto_preset_combo.currentData()
        presets: dict[str, dict[str, object]] = {
            "conservative": {
                "seed_percentile": 94.0,
                "seed_pixel_percent": 0.25,
                "dilate_iterations": 1,
                "bg1_percentile": 12.0,
                "bg1_bins": 0,
                "grow_diff_pct": 20.0,
                "grow_hist_pct": 30.0,
                "grow_force_pct": -1.0,
                "grow_limit": 8000,
                "bg2_percentile": 20.0,
                "bg2_bins": 0,
                "addition_support_percentile": 82.0,
                "protect_small_original": True,
                "small_component_guard": 140,
            },
            "aggressive": {
                "seed_percentile": 88.0,
                "seed_pixel_percent": 0.8,
                "dilate_iterations": 1,
                "bg1_percentile": 9.0,
                "bg1_bins": 0,
                "grow_diff_pct": 32.0,
                "grow_hist_pct": 22.0,
                "grow_force_pct": -1.0,
                "grow_limit": 22000,
                "bg2_percentile": 13.0,
                "bg2_bins": 0,
                "addition_support_percentile": 72.0,
                "protect_small_original": True,
                "small_component_guard": 110,
            },
            "balanced": {
                "seed_percentile": 92.0,
                "seed_pixel_percent": 0.45,
                "dilate_iterations": 1,
                "bg1_percentile": 10.0,
                "bg1_bins": 0,
                "grow_diff_pct": 26.0,
                "grow_hist_pct": 28.0,
                "grow_force_pct": -1.0,
                "grow_limit": 14000,
                "bg2_percentile": 16.0,
                "bg2_bins": 0,
                "addition_support_percentile": 78.0,
                "protect_small_original": True,
                "small_component_guard": 120,
            },
        }
        return presets.get(preset, presets["balanced"]).copy()

    def _quick_auto_is_running(self) -> bool:
        return self.quick_auto_thread is not None and self.quick_auto_thread.isRunning()

    def _set_quick_auto_busy(self, busy: bool) -> None:
        if hasattr(self, "quick_auto_preset_combo"):
            self.quick_auto_preset_combo.setEnabled(not busy)
        if hasattr(self, "quick_auto_btn"):
            self.quick_auto_btn.setEnabled(not busy)
        if hasattr(self, "quick_auto_stack_btn"):
            self.quick_auto_stack_btn.setEnabled(not busy)
        if hasattr(self, "quick_auto_cancel_btn"):
            self.quick_auto_cancel_btn.setEnabled(busy and not self.quick_auto_cancel_event.is_set())
        if hasattr(self, "quick_auto_revert_btn"):
            self.quick_auto_revert_btn.setEnabled((not busy) and self._quick_auto_snapshot_masks is not None)
        for name in (
            "quick_auto_act",
            "quick_auto_stack_act",
            "quick_auto_revert_act",
        ):
            action = getattr(self, name, None)
            if action is not None:
                action.setEnabled(not busy)
        action = getattr(self, "quick_auto_cancel_act", None)
        if action is not None:
            action.setEnabled(busy and not self.quick_auto_cancel_event.is_set())

    def _stage_quick_auto_snapshot(self, label: str) -> bool:
        if not self._ensure_masks():
            return False
        if self.model.masks is None:
            return False
        self._pending_quick_auto_snapshot_masks = self.model.masks.copy()
        self._pending_quick_auto_snapshot_index = self.model.index
        self._pending_quick_auto_snapshot_label = label
        return True

    def _clear_staged_quick_auto_snapshot(self) -> None:
        self._pending_quick_auto_snapshot_masks = None
        self._pending_quick_auto_snapshot_index = 0
        self._pending_quick_auto_snapshot_label = ""

    def _commit_staged_quick_auto_snapshot(self) -> None:
        if self._pending_quick_auto_snapshot_masks is None:
            return
        self._quick_auto_snapshot_masks = self._pending_quick_auto_snapshot_masks
        self._quick_auto_snapshot_index = self._pending_quick_auto_snapshot_index
        self._quick_auto_snapshot_label = self._pending_quick_auto_snapshot_label
        self._clear_staged_quick_auto_snapshot()

    def _cleanup_quick_auto_thread(self) -> None:
        self.quick_auto_thread = None
        self._active_quick_auto_request_id = None
        self._active_quick_auto_mode = None
        self._pending_quick_auto_undo_mask = None
        self.quick_auto_cancel_event.clear()
        self._set_quick_auto_busy(False)

    def _start_quick_auto_job(
        self,
        *,
        mode: str,
        indices: list[int],
        label: str,
        params: dict[str, object],
    ) -> bool:
        if self._quick_auto_is_running():
            return False
        if not indices:
            return False
        if not self._stage_quick_auto_snapshot(label):
            return False
        if self.model.data is None or self.model.masks is None:
            self._clear_staged_quick_auto_snapshot()
            return False
        request_id = int(self._quick_auto_request_seq) + 1
        self._quick_auto_request_seq = request_id
        self._active_quick_auto_request_id = request_id
        self._active_quick_auto_mode = mode
        self._pending_quick_auto_undo_mask = (
            self.model.masks.copy() if mode == "single" else None
        )
        self.quick_auto_cancel_event.clear()
        thread = QuickAutoThread(
            data_stack=np.array(self.model.data, copy=True, order="C"),
            mask_stack=np.array(self.model.masks, copy=True, order="C"),
            params=params,
            request_id=request_id,
            image_revision=int(self.model.image_revision),
            mask_revision=int(self.model.mask_revision),
            mode=mode,
            indices=indices,
            cancel_event=self.quick_auto_cancel_event,
        )
        thread.finished.connect(thread.deleteLater)
        thread.succeeded.connect(self._quick_auto_finished)
        thread.failed.connect(self._quick_auto_failed)
        thread.cancelled.connect(self._quick_auto_cancelled)
        thread.progress.connect(self._quick_auto_progress)
        self.quick_auto_thread = thread
        self._set_quick_auto_busy(True)
        if mode == "stack":
            self.statusBar().showMessage(
                f"Quick auto stack running in background for {len(indices)} slice(s)..."
            )
        else:
            self.statusBar().showMessage(
                f"Quick auto running in background for slice {indices[0] + 1}..."
            )
        thread.start()
        return True

    def _cancel_quick_auto(self) -> None:
        if not self._quick_auto_is_running():
            self.statusBar().showMessage("No quick auto run is active.")
            return
        if self.quick_auto_cancel_event.is_set():
            return
        self.quick_auto_cancel_event.set()
        self._set_quick_auto_busy(True)
        label = "stack" if self._active_quick_auto_mode == "stack" else "current slice"
        self.statusBar().showMessage(f"Cancelling quick auto for {label}...")

    def _quick_auto_progress(self, payload: dict) -> None:
        request_id = int(payload.get("request_id", -1))
        if request_id != getattr(self, "_active_quick_auto_request_id", None):
            return
        processed = int(payload.get("processed", 0))
        total = int(payload.get("total", 0))
        slice_index = int(payload.get("slice_index", -1))
        self.statusBar().showMessage(
            f"Quick auto stack running in background... {processed}/{total} slice(s) "
            f"(latest slice {slice_index + 1})."
        )

    def _quick_auto_cancelled(self, payload: dict) -> None:
        try:
            request_id = int(payload.get("request_id", -1))
            if request_id != getattr(self, "_active_quick_auto_request_id", None):
                return
            self._clear_staged_quick_auto_snapshot()
            processed = int(payload.get("processed", 0))
            total = int(payload.get("total", 0))
            mode = str(payload.get("mode", "single"))
            elapsed = float(payload.get("elapsed_sec", 0.0))
            if mode == "stack":
                self.statusBar().showMessage(
                    f"Quick auto stack cancelled after {processed}/{total} slice(s) in {elapsed:.2f}s. "
                    "No changes were applied."
                )
            else:
                self.statusBar().showMessage(
                    f"Quick auto cancelled in {elapsed:.2f}s. No changes were applied."
                )
        finally:
            self._cleanup_quick_auto_thread()

    def _quick_auto_failed(self, payload: dict) -> None:
        try:
            request_id = int(payload.get("request_id", -1))
            if request_id != getattr(self, "_active_quick_auto_request_id", None):
                return
            self._clear_staged_quick_auto_snapshot()
            error = str(payload.get("error", "unknown error"))
            self.statusBar().showMessage(f"Quick auto failed: {error}")
        finally:
            self._cleanup_quick_auto_thread()

    def _quick_auto_finished(self, payload: dict) -> None:
        try:
            request_id = int(payload.get("request_id", -1))
            if request_id != getattr(self, "_active_quick_auto_request_id", None):
                return
            if (
                int(self.model.image_revision) != int(payload.get("image_revision", -1))
                or int(self.model.mask_revision) != int(payload.get("mask_revision", -1))
            ):
                self._clear_staged_quick_auto_snapshot()
                self.statusBar().showMessage(
                    "Quick auto result discarded: image or mask changed while background quick auto was running."
                )
                return
            mode = str(payload.get("mode", "single"))
            if mode == "stack":
                self._apply_quick_auto_stack_result(payload)
            else:
                self._apply_quick_auto_single_result(payload)
        finally:
            self._cleanup_quick_auto_thread()

    def _apply_quick_auto_single_result(self, payload: dict) -> None:
        slice_index = int(payload.get("slice_index", self.model.index))
        result_mask = np.asarray(payload.get("mask"), dtype=np.uint8)
        metrics = {
            "before_pixels": int(payload.get("before_pixels", 0)),
            "after_pixels": int(payload.get("after_pixels", 0)),
            "changed": bool(payload.get("changed", False)),
        }
        if metrics["changed"]:
            if self._pending_quick_auto_undo_mask is not None:
                self._push_undo("quick_auto", mask=self._pending_quick_auto_undo_mask)
            self.model.set_mask(result_mask, slice_idx=slice_index)
            self._commit_staged_quick_auto_snapshot()
        else:
            self._clear_staged_quick_auto_snapshot()
        self._update_view()
        self._post_quick_auto_quality_gate(
            metrics,
            context_label=f"slice {slice_index + 1}",
            elapsed_sec=float(payload.get("elapsed_sec", 0.0)),
        )

    def _apply_quick_auto_stack_result(self, payload: dict) -> None:
        masks = np.asarray(payload.get("masks"), dtype=np.uint8)
        indices = [int(idx) for idx in payload.get("indices", [])]
        metrics_by_slice = list(payload.get("metrics_by_slice", []))
        changed = bool(payload.get("changed", False))
        self.model.replace_masks(masks, dirty=True)
        if indices:
            target_index = max(0, min(self.model.n_slices - 1, indices[-1]))
            self.model.index = target_index
            if self.slider.isEnabled():
                self.slider.setValue(target_index)
        self._update_view()
        if changed:
            self._commit_staged_quick_auto_snapshot()
        else:
            self._clear_staged_quick_auto_snapshot()
        self._post_quick_auto_stack_quality_gate(
            metrics_by_slice,
            indices,
            elapsed_sec=float(payload.get("elapsed_sec", 0.0)),
        )

    def _restore_quick_auto_snapshot(self) -> bool:
        if self._quick_auto_is_running():
            self.statusBar().showMessage("Cancel the active quick auto run before reverting.")
            return False
        if self._quick_auto_snapshot_masks is None:
            QMessageBox.information(
                self,
                "Revert Auto Snapshot",
                "No snapshot available. Run Quick Auto first.",
            )
            return False
        self.model.replace_masks(self._quick_auto_snapshot_masks.copy(), dirty=True)
        target_index = max(0, min(self.model.n_slices - 1, self._quick_auto_snapshot_index))
        self.model.index = target_index
        if self.slider.isEnabled():
            self.slider.setValue(target_index)
        self._update_view()
        self.statusBar().showMessage(
            f"Reverted to snapshot: {self._quick_auto_snapshot_label}"
        )
        return True

    def _evaluate_quick_auto_quality(self, metrics: dict[str, int]) -> tuple[bool, str]:
        before = int(metrics.get("before_pixels", 0))
        after = int(metrics.get("after_pixels", 0))
        area = int(self.model.get_mask().size)
        preset = self.quick_auto_preset_combo.currentData()
        gate_cfg = {
            "conservative": {"max_growth_pct": 120.0, "max_area_if_empty_pct": 8.0, "max_area_pct": 35.0},
            "balanced": {"max_growth_pct": 180.0, "max_area_if_empty_pct": 12.0, "max_area_pct": 45.0},
            "aggressive": {"max_growth_pct": 260.0, "max_area_if_empty_pct": 18.0, "max_area_pct": 58.0},
        }.get(preset, {"max_growth_pct": 180.0, "max_area_if_empty_pct": 12.0, "max_area_pct": 45.0})

        area_pct = after * 100.0 / float(max(area, 1))
        if area_pct > gate_cfg["max_area_pct"]:
            return (
                False,
                (
                    f"Foreground coverage too large: {area_pct:.1f}% "
                    f"(limit {gate_cfg['max_area_pct']:.1f}%)."
                ),
            )

        if before > 0:
            growth_pct = (after - before) * 100.0 / float(before)
            if growth_pct > gate_cfg["max_growth_pct"]:
                return (
                    False,
                    (
                        f"Foreground grew too much: {before} -> {after} "
                        f"({growth_pct:.1f}% increase, limit {gate_cfg['max_growth_pct']:.1f}%)."
                    ),
                )
        else:
            filled_pct = after * 100.0 / float(max(area, 1))
            if filled_pct > gate_cfg["max_area_if_empty_pct"]:
                return (
                    False,
                    (
                        f"Started from empty mask but filled too much area: "
                        f"{filled_pct:.1f}% (limit {gate_cfg['max_area_if_empty_pct']:.1f}%)."
                    ),
                )
        return True, "Quality gate passed."

    def _post_quick_auto_quality_gate(
        self,
        metrics: dict[str, int],
        context_label: str,
        elapsed_sec: float | None = None,
    ) -> None:
        is_ok, message = self._evaluate_quick_auto_quality(metrics)
        before = int(metrics.get("before_pixels", 0))
        after = int(metrics.get("after_pixels", 0))
        elapsed_text = "" if elapsed_sec is None else f" in {elapsed_sec:.2f}s"
        if is_ok:
            self.statusBar().showMessage(
                f"Quick auto done ({context_label}){elapsed_text} | pixels {before} -> {after} | gate passed."
            )
            return

        ret = QMessageBox.question(
            self,
            "Quick Auto Quality Gate",
            (
                f"{message}\n\n"
                "The result may be over-expanded.\n"
                "Revert to the pre-run snapshot?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if ret == QMessageBox.Yes:
            self._restore_quick_auto_snapshot()
            return
        self.statusBar().showMessage(
            f"Quick auto kept despite gate warning{elapsed_text} | pixels {before} -> {after}."
        )

    def _post_quick_auto_stack_quality_gate(
        self,
        metrics_by_slice: list[dict[str, object]],
        indices: list[int],
        *,
        elapsed_sec: float | None = None,
    ) -> None:
        flagged: list[int] = []
        for metrics in metrics_by_slice:
            eval_metrics = {
                "before_pixels": int(metrics.get("before_pixels", 0)),
                "after_pixels": int(metrics.get("after_pixels", 0)),
                "changed": bool(metrics.get("changed", False)),
            }
            is_ok, _message = self._evaluate_quick_auto_quality(eval_metrics)
            if not is_ok:
                flagged.append(int(metrics.get("slice_index", -1)) + 1)
        elapsed_text = "" if elapsed_sec is None else f" in {elapsed_sec:.2f}s"
        if flagged:
            preview = ", ".join(str(x) for x in flagged[:15])
            suffix = " ..." if len(flagged) > 15 else ""
            ret = QMessageBox.question(
                self,
                "Quick Auto Stack Quality Gate",
                (
                    f"Finished {len(indices)} slice(s){elapsed_text}. "
                    f"{len(flagged)} slice(s) exceeded quality gate:\n"
                    f"{preview}{suffix}\n\n"
                    "Revert all stack changes back to the pre-run snapshot?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ret == QMessageBox.Yes:
                self._restore_quick_auto_snapshot()
                return
            self.statusBar().showMessage(
                f"Quick auto stack kept with warnings on {len(flagged)} slice(s){elapsed_text}."
            )
            return
        changed_count = sum(1 for metrics in metrics_by_slice if bool(metrics.get("changed")))
        self.statusBar().showMessage(
            f"Quick auto stack finished for {len(indices)} slice(s){elapsed_text}; "
            f"{changed_count} slice(s) changed and no gate warnings."
        )

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

    def _open_comparison_dialog(self) -> None:
        """Open the strategy comparison dialog."""
        dialog = ComparisonDialog(self)
        dialog.exec_()

    def _open_validation_dialog(self) -> None:
        """Open the validation viewer dialog."""
        dialog = ValidationDialog(self)
        dialog.exec_()

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
        choice, ok = QInputDialog.getItem(
            self,
            "Frangi Filter",
            "Black ridges?",
            ["True", "False"],
            0,
            False,
        )
        if not ok:
            return
        self.script_frangi_filter(start, end, step, thresh, choice == "True")

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
        choice, ok = QInputDialog.getItem(
            self,
            "Sato Filter",
            "Black ridges?",
            ["True", "False"],
            0,
            False,
        )
        if not ok:
            return
        self.script_sato_filter(start, end, step, thresh, choice == "True")

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
        choice, ok = QInputDialog.getItem(
            self,
            "Meijering Filter",
            "Black ridges?",
            ["True", "False"],
            0,
            False,
        )
        if not ok:
            return
        self.script_meijering_filter(start, end, step, thresh, choice == "True")

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
