# Legacy version of the ZStackAnnotationTool GUI.
# This file is kept for historical reference and is not used by the application.

import sys
import os
import glob
import numpy as np
from skimage.morphology import diamond, binary_erosion, binary_dilation
from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import tifffile
from skimage import measure
from skimage.morphology import (
    binary_dilation, 
    binary_erosion, 
    remove_small_objects, 
    diamond
)

class MplCanvas(FigureCanvasQTAgg):
    """A helper class to integrate a matplotlib Figure into PyQt5."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)
        fig.tight_layout()

class ZStackAnnotationTool(QtWidgets.QMainWindow):
    def __init__(self):
        super(ZStackAnnotationTool, self).__init__()
        
        self.setWindowTitle("Z-Stack Annotation Tool (Undo/Redo + new random seeds feature)")
        self.resize(1200, 800)
        
        # ------------------- Main Layout -------------------
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # ===== Top row: open files/folders & load
        file_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(file_layout)
        
        self.open_image_button = QtWidgets.QPushButton("Open Image...")
        self.open_annotation_button = QtWidgets.QPushButton("Open Annotations...")
        self.open_folder_button = QtWidgets.QPushButton("Open Folder of Masks...")
        self.load_button = QtWidgets.QPushButton("Load")
        self.close_current_button = QtWidgets.QPushButton("Close Current Data")
        
        file_layout.addWidget(self.open_image_button)
        file_layout.addWidget(self.open_annotation_button)
        file_layout.addWidget(self.open_folder_button)
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.close_current_button)
        
        # ===== Central Canvas
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        main_layout.addWidget(self.canvas)
        
        # ===== Bottom tool area
        tool_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(tool_layout)
        
        # == Z slice control
        slice_layout = QtWidgets.QVBoxLayout()
        slice_label = QtWidgets.QLabel("Z Slice")
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.setEnabled(False)
        
        slice_button_layout = QtWidgets.QHBoxLayout()
        self.prev_slice_button = QtWidgets.QPushButton("Prev Slice")
        self.next_slice_button = QtWidgets.QPushButton("Next Slice")
        slice_button_layout.addWidget(self.prev_slice_button)
        slice_button_layout.addWidget(self.next_slice_button)
        
        slice_layout.addWidget(slice_label)
        slice_layout.addWidget(self.slice_slider)
        slice_layout.addLayout(slice_button_layout)
        
        self.info_label = QtWidgets.QLabel("No image loaded")
        slice_layout.addWidget(self.info_label)
        
        tool_layout.addLayout(slice_layout)
        
        # == Filtering
        filter_layout = QtWidgets.QVBoxLayout()
        filter_label = QtWidgets.QLabel("Min area (pixel)")
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("e.g. 100")
        self.filter_apply_button = QtWidgets.QPushButton("Apply Filter")
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_edit)
        filter_layout.addWidget(self.filter_apply_button)
        tool_layout.addLayout(filter_layout)
        
        # == Morphology with adjustable strength
        morph_layout = QtWidgets.QVBoxLayout()
        morph_layout.addWidget(QtWidgets.QLabel("Morphology (Strength)"))
        self.morph_strength_edit = QtWidgets.QLineEdit()
        self.morph_strength_edit.setPlaceholderText("default=1")
        self.morph_strength_edit.setText("1")
        self.erode_button = QtWidgets.QPushButton("Erode")
        self.dilate_button = QtWidgets.QPushButton("Dilate")
        morph_layout.addWidget(self.morph_strength_edit)
        morph_layout.addWidget(self.erode_button)
        morph_layout.addWidget(self.dilate_button)
        
        # Undo & Redo按钮
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.redo_button = QtWidgets.QPushButton("Redo")
        morph_layout.addWidget(self.undo_button)
        morph_layout.addWidget(self.redo_button)
        
        tool_layout.addLayout(morph_layout)

        # == Mask background removal based on relative intensity
        bg_filter_layout = QtWidgets.QVBoxLayout()
        bg_filter_layout.addWidget(QtWidgets.QLabel("Remove BG Pixels (%)"))
        self.bg_percentile_edit = QtWidgets.QLineEdit()
        self.bg_percentile_edit.setPlaceholderText("e.g. 20 means remove bottom 20%")
        self.bg_filter_button = QtWidgets.QPushButton("Apply BG Filter")
        bg_filter_layout.addWidget(self.bg_percentile_edit)
        bg_filter_layout.addWidget(self.bg_filter_button)
        tool_layout.addLayout(bg_filter_layout)
            
        # == Contrast
        stretch_layout = QtWidgets.QVBoxLayout()
        stretch_label = QtWidgets.QLabel("Histogram Stretch (%)")
        self.stretch_edit = QtWidgets.QLineEdit()
        self.stretch_edit.setPlaceholderText("e.g. 1")
        self.stretch_button = QtWidgets.QPushButton("Apply Stretch")
        self.reset_stretch_button = QtWidgets.QPushButton("Reset Contrast")
        stretch_layout.addWidget(stretch_label)
        stretch_layout.addWidget(self.stretch_edit)
        stretch_layout.addWidget(self.stretch_button)
        stretch_layout.addWidget(self.reset_stretch_button)
        tool_layout.addLayout(stretch_layout)
        
        # == Sub-stack range
        substack_layout = QtWidgets.QVBoxLayout()
        substack_label = QtWidgets.QLabel("Sub-stack Range")
        self.substack_start_edit = QtWidgets.QLineEdit()
        self.substack_start_edit.setPlaceholderText("Start z e.g. 10")
        self.substack_end_edit = QtWidgets.QLineEdit()
        self.substack_end_edit.setPlaceholderText("End z e.g. 40")
        substack_layout.addWidget(substack_label)
        substack_layout.addWidget(self.substack_start_edit)
        substack_layout.addWidget(self.substack_end_edit)
        tool_layout.addLayout(substack_layout)
        
        # == Selection & deletion
        delete_layout = QtWidgets.QVBoxLayout()
        delete_layout.addWidget(QtWidgets.QLabel("Delete mask"))
        self.select_mode_button = QtWidgets.QPushButton("Select Mode")
        delete_layout.addWidget(self.select_mode_button)
        tool_layout.addLayout(delete_layout)
        
        # == 新增随机“播种”功能（UI文字改成40%）
        random_seed_layout = QtWidgets.QVBoxLayout()
        random_seed_label = QtWidgets.QLabel("Random Seeds (top 40%)")  # <--- 仅显示文字的改动
        self.random_seed_button = QtWidgets.QPushButton("Random Seeds")
        random_seed_layout.addWidget(random_seed_label)
        random_seed_layout.addWidget(self.random_seed_button)
        tool_layout.addLayout(random_seed_layout)
        
        # == 基于亮度分布的梯度扩展 + 新增第二阈值
        gradient_layout = QtWidgets.QVBoxLayout()
        gradient_label = QtWidgets.QLabel("Gradient Expand (%)")
        self.gradient_expand_edit = QtWidgets.QLineEdit()
        self.gradient_expand_edit.setPlaceholderText("e.g. 20")
        
        # -------- 新增的第二阈值输入框 --------
        slice_thresh_label = QtWidgets.QLabel("Min Slice Intensity(%)")
        self.gradient_slice_thresh_edit = QtWidgets.QLineEdit()
        self.gradient_slice_thresh_edit.setPlaceholderText("e.g. 30")
        
        self.gradient_expand_button = QtWidgets.QPushButton("Gradient Expand")
        gradient_layout.addWidget(gradient_label)
        gradient_layout.addWidget(self.gradient_expand_edit)
        gradient_layout.addWidget(slice_thresh_label)
        gradient_layout.addWidget(self.gradient_slice_thresh_edit)
        gradient_layout.addWidget(self.gradient_expand_button)
        tool_layout.addLayout(gradient_layout)
        
        # == Right side
        right_buttons_layout = QtWidgets.QVBoxLayout()
        
        # Update Masks
        self.update_masks_button = QtWidgets.QPushButton("Update Masks")
        self.update_masks_button.setToolTip("Re-check entire Z stack and re-label all slices.")
        right_buttons_layout.addWidget(self.update_masks_button)
        
        # Save
        self.save_button = QtWidgets.QPushButton("Save Annotation + Sub-stack")
        right_buttons_layout.addWidget(self.save_button)
        
        # Close Window (red button)
        self.close_window_button = QtWidgets.QPushButton("Close Window")
        self.close_window_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        right_buttons_layout.addWidget(self.close_window_button)
        
        tool_layout.addLayout(right_buttons_layout)
        
        # ----- State Variables -----
        self.image_path = None
        self.annotation_paths = []
        
        self.original_zstack = None
        self.annotation_zstack = None
        
        self.raw_display_zstack = None
        self.stretched_display_zstack = None
        self.use_stretched = False
        
        self.current_z = 0
        self.select_mode = False
        
        # For zoom/pan
        self._zoom_scale = 1.1
        self._panning = False
        self._pan_start_xy = (0, 0)
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        
        # For rectangle-based deletion
        self._dragging_rect = False
        self._rect_patch = None
        self._rect_start = (0, 0)
        
        self._closing = False  # for closeEvent control
        
        # Undo/Redo Stack
        self.undo_stack = []
        self.redo_stack = []
        self.MAX_STACK_SIZE = 5
        
        # ----- Signals & Slots -----
        self.open_image_button.clicked.connect(self.choose_image_file)
        self.open_annotation_button.clicked.connect(self.choose_annotation_files)
        self.open_folder_button.clicked.connect(self.choose_annotation_folder)
        
        self.load_button.clicked.connect(self.load_data)
        self.close_current_button.clicked.connect(self.close_current)
        
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        
        self.prev_slice_button.clicked.connect(self.prev_slice)
        self.next_slice_button.clicked.connect(self.next_slice)
        
        self.filter_apply_button.clicked.connect(self.apply_filter_current_slice)
        self.erode_button.clicked.connect(self.erode_current_slice)
        self.dilate_button.clicked.connect(self.dilate_current_slice)
        
        self.stretch_button.clicked.connect(self.apply_stretch)
        self.reset_stretch_button.clicked.connect(self.reset_stretch)
        
        self.select_mode_button.clicked.connect(self.toggle_select_mode)
        
        self.update_masks_button.clicked.connect(self.update_masks)
        self.save_button.clicked.connect(self.save_annotation)
        
        self.close_window_button.clicked.connect(self._on_close_window)
        
        self.bg_filter_button.clicked.connect(self.remove_bg_pixels_current_slice)
        
        self.undo_button.clicked.connect(self.undo_last_operation)
        self.redo_button.clicked.connect(self.redo_operation)
        
        # 新增随机播种按钮
        self.random_seed_button.clicked.connect(self.random_seeds_current_slice)
        
        # 新增梯度扩展按钮（添加第二阈值逻辑）
        self.gradient_expand_button.clicked.connect(self.gradient_expand_current_slice)
        
        # Matplotlib events
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        
        # Listen for arrow keys
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    # ----------------------------------------------------------------
    #  KEY PRESS: z => Undo, x => Redo, 1/2/3/4/5 => various ops
    # ----------------------------------------------------------------
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        line_edits = [
            self.morph_strength_edit,
            self.bg_percentile_edit,
            self.filter_edit,
            self.stretch_edit,
            self.substack_start_edit,
            self.substack_end_edit,
            self.gradient_expand_edit,
            self.gradient_slice_thresh_edit,  # 新增行
        ]
        if self.focusWidget() in line_edits:
            # 如果正在输入框中，则不拦截这些快捷键
            super(ZStackAnnotationTool, self).keyPressEvent(event)
            return
        
        key = event.key()
        if key == QtCore.Qt.Key_1:
            self.dilate_current_slice()
        elif key == QtCore.Qt.Key_2:
            self.erode_current_slice()
        elif key == QtCore.Qt.Key_3:
            self.remove_bg_pixels_current_slice()
        elif key == QtCore.Qt.Key_4:
            self.apply_filter_current_slice()
        elif key == QtCore.Qt.Key_5:
            self.gradient_expand_current_slice()  # <--- 新增快捷键 5
        elif key == QtCore.Qt.Key_Left:
            self.prev_slice()
        elif key == QtCore.Qt.Key_Right:
            self.next_slice()
        elif key == QtCore.Qt.Key_Z:
            self.undo_last_operation()
        elif key == QtCore.Qt.Key_X:
            self.redo_operation()
        elif key == QtCore.Qt.Key_S:
            self.random_seeds_current_slice()
        else:
            super(ZStackAnnotationTool, self).keyPressEvent(event)
    
    # ----------------------------------------------------------------
    #  Undo/Redo stack logic
    # ----------------------------------------------------------------
    def push_undo_stack(self):
        """在对annotation_zstack做任何新操作前调用。保存当前状态到 undo_stack，并清空redo_stack。"""
        if self.annotation_zstack is not None:
            backup = np.copy(self.annotation_zstack)
            self.undo_stack.append(backup)
            # 超过最大步数则移除最早的
            while len(self.undo_stack) > self.MAX_STACK_SIZE:
                self.undo_stack.pop(0)
            # 一旦有新操作，redo_stack 就作废
            self.redo_stack.clear()

    def undo_last_operation(self):
        """从undo_stack里弹出一个版本，推入redo_stack，并恢复到它。"""
        if len(self.undo_stack) == 0:
            QtWidgets.QMessageBox.information(self, "Info", "没有可撤销的操作。")
            return
        
        # 当前状态 -> redo
        current_backup = np.copy(self.annotation_zstack)
        self.redo_stack.append(current_backup)
        while len(self.redo_stack) > self.MAX_STACK_SIZE:
            self.redo_stack.pop(0)

        # 从undo弹出 => 恢复
        last = self.undo_stack.pop()
        self.annotation_zstack = last
        self.update_display()
        self.canvas.draw()
    
    def redo_operation(self):
        """从redo_stack里弹出一个版本，推入undo_stack，并恢复到它。"""
        if len(self.redo_stack) == 0:
            QtWidgets.QMessageBox.information(self, "Info", "没有可重做的操作。")
            return
        
        # 当前状态 -> undo
        current_backup = np.copy(self.annotation_zstack)
        self.undo_stack.append(current_backup)
        while len(self.undo_stack) > self.MAX_STACK_SIZE:
            self.undo_stack.pop(0)
        
        # 从redo弹出 => 恢复
        last = self.redo_stack.pop()
        self.annotation_zstack = last
        self.update_display()
        self.canvas.draw()

    # ----------------------------------------------------------------
    #  Basic slice nav
    # ----------------------------------------------------------------
    def prev_slice(self):
        if self.original_zstack is not None:
            new_z = max(0, self.current_z - 1)
            self.slice_slider.setValue(new_z)
    
    def next_slice(self):
        if self.original_zstack is not None:
            max_z = self.original_zstack.shape[0] - 1
            new_z = min(max_z, self.current_z + 1)
            self.slice_slider.setValue(new_z)

    # ----------------------------------------------------------------
    #  Overriding closeEvent
    # ----------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent):
        if not self._closing:
            QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                "Please use the red 'Close Window' button to close this window."
            )
            event.ignore()
        else:
            event.accept()

    def _on_close_window(self):
        self._closing = True
        self.close()
        app = QtWidgets.QApplication.instance()
        if app:
            app.quit()

    # ----------------------------------------------------------------
    #  Load / Close data
    # ----------------------------------------------------------------
    def choose_annotation_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Folder of Masks", "")
        if folder:
            mask_files = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff"))
            mask_files.sort()
            if mask_files:
                self.annotation_paths.extend(mask_files)
            else:
                QtWidgets.QMessageBox.information(self, "Info", f"No .tif/.tiff found in {folder}.")

    def choose_image_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)"
        )
        if path:
            self.image_path = path
    
    def choose_annotation_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open Annotations",
            "",
            "Annotation Files (*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)"
        )
        if paths:
            self.annotation_paths.extend(paths)

    def load_data(self):
        if not self.image_path or not os.path.isfile(self.image_path):
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid image file.")
            return
        
        if len(self.annotation_paths) == 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No annotation file chosen or found.")
            return
        
        try:
            self.original_zstack = tifffile.imread(self.image_path)
            Z = self.original_zstack.shape[0]
            if Z < 1:
                raise ValueError("Original image must have at least one slice in Z.")
            
            combined_anno = None
            for anno_path in self.annotation_paths:
                if not os.path.isfile(anno_path):
                    print(f"Warning: file not exist: {anno_path}")
                    continue
                temp_anno = tifffile.imread(anno_path)
                if temp_anno.shape[0] != Z:
                    raise ValueError(f"Annotation {anno_path} shape mismatch with original (Z dimension).")
                
                temp_anno_bin = (temp_anno > 0).astype(np.uint8)
                if combined_anno is None:
                    combined_anno = temp_anno_bin
                else:
                    combined_anno = np.logical_or(combined_anno, temp_anno_bin).astype(np.uint8)
            
            if combined_anno is None:
                raise ValueError("No valid annotation loaded.")
            
            self.annotation_zstack = combined_anno.astype(np.int32)
            
            self.slice_slider.setMaximum(Z - 1)
            self.slice_slider.setEnabled(True)
            self.current_z = 0
            self.slice_slider.setValue(0)
            
            # Prepare display for original
            self.raw_display_zstack = []
            for z_idx in range(Z):
                slc_8bit = self._extract_slice(self.original_zstack, z_idx)
                self.raw_display_zstack.append(slc_8bit)
            self.raw_display_zstack = np.stack(self.raw_display_zstack, axis=0)
            
            self.stretched_display_zstack = self.raw_display_zstack.copy()
            self.use_stretched = False
            
            # 清空撤销 / 重做
            self.undo_stack.clear()
            self.redo_stack.clear()
            
            self.canvas.axes.set_xlim(None)
            self.canvas.axes.set_ylim(None)
            
            self.update_display()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load: {e}")
    
    def close_current(self):
        self.original_zstack = None
        self.annotation_zstack = None
        self.raw_display_zstack = None
        self.stretched_display_zstack = None
        
        self.image_path = None
        self.annotation_paths = []
        
        self.current_z = 0
        self.slice_slider.setValue(0)
        self.slice_slider.setEnabled(False)
        self.info_label.setText("No image loaded")
        
        self.canvas.axes.clear()
        self.canvas.draw()

    # ----------------------------------------------------------------
    #  Display / Update
    # ----------------------------------------------------------------
    def on_slice_changed(self, value):
        self.current_z = value
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        self.update_display()
        
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()
    
    def update_display(self):
        if self.original_zstack is None or self.annotation_zstack is None:
            return
        
        z = self.current_z
        
        if self.use_stretched:
            raw_slice = self.stretched_display_zstack[z]
        else:
            raw_slice = self.raw_display_zstack[z]
        
        mask_slice = self.annotation_zstack[z]
        
        labeled_slice = measure.label(mask_slice, connectivity=1, background=0)
        props = measure.regionprops(labeled_slice)
        num_masks = len(props)
        total_pixels = sum([p.area for p in props])
        
        self.info_label.setText(f"Z={z} | #Masks: {num_masks}, Total pixels: {total_pixels}")
        
        self.canvas.axes.clear()
        self.canvas.axes.imshow(raw_slice, cmap='gray', vmin=0, vmax=255)
        
        overlay = (mask_slice > 0).astype(np.uint8)
        alpha_val = 0.4
        color_overlay = np.zeros((overlay.shape[0], overlay.shape[1], 4), dtype=np.float32)
        color_overlay[overlay == 1, 0] = 1.0   # red
        color_overlay[overlay == 1, 3] = alpha_val
        
        self.canvas.axes.imshow(color_overlay)
        
        self.canvas.draw()
    
    def _extract_slice(self, volume, z):
        if volume.ndim == 3:
            slc = volume[z, :, :]
        elif volume.ndim == 4:
            slc = volume[z, ...]
            # 如果是多通道，简单处理为取RGB前三通道的平均
            if slc.shape[-1] in (3,4):
                slc = np.mean(slc[..., :3], axis=-1)
            else:
                slc = np.mean(slc, axis=-1)
        else:
            raise ValueError("Volume dimension not supported.")
        return self._normalize_to_8bit(slc)
    
    def _normalize_to_8bit(self, arr):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn)/(mx - mn)*255.0
        return arr.astype(np.uint8)

    # ----------------------------------------------------------------
    #  Key operations: Filter/Erode/Dilate/Remove BG => push_undo_stack first
    # ----------------------------------------------------------------
    def apply_filter_current_slice(self):
        if self.annotation_zstack is None:
            return
        
        self.push_undo_stack()
        
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        threshold_str = self.filter_edit.text()
        try:
            threshold = int(threshold_str)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Please input a valid integer for min area.")
            return
        
        z = self.current_z
        labeled_slice = measure.label(self.annotation_zstack[z], connectivity=1, background=0)
        filtered = remove_small_objects(labeled_slice, min_size=threshold)
        
        self.annotation_zstack[z] = filtered
        self.update_display()

        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()

    def erode_current_slice(self):
        if self.annotation_zstack is None:
            return
        
        self.push_undo_stack()  # 备份
        
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        try:
            strength = int(self.morph_strength_edit.text())
        except:
            strength = 1
        selem = diamond(strength)
        z = self.current_z
        binary_slice = (self.annotation_zstack[z] > 0)
        eroded = binary_erosion(binary_slice, footprint=selem)
        self.annotation_zstack[z] = measure.label(eroded, connectivity=1)
        self.update_display()
        
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()

    def dilate_current_slice(self):
        if self.annotation_zstack is None:
            return
        
        self.push_undo_stack()  # 备份

        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        try:
            strength = int(self.morph_strength_edit.text())
        except:
            strength = 1
        selem = diamond(strength)
        z = self.current_z
        
        # 与 erode 的逻辑类似，先转二值，再做膨胀，然后重新 label
        binary_slice = (self.annotation_zstack[z] > 0)
        dilated = binary_dilation(binary_slice, footprint=selem)
        self.annotation_zstack[z] = measure.label(dilated, connectivity=1, background=0)
        
        self.update_display()
        
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()
        
    def remove_bg_pixels_current_slice(self):
        if self.annotation_zstack is None or self.original_zstack is None:
            return
        
        self.push_undo_stack()  # 备份

        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()

        try:
            percentile = float(self.bg_percentile_edit.text())
            if percentile <= 0 or percentile >= 100:
                raise ValueError
        except:
            QtWidgets.QMessageBox.warning(self, "Error", "请输入有效的百分位数 (0-100之间).")
            return

        z = self.current_z
        raw_slice = self._extract_slice(self.original_zstack, z)
        mask_slice = self.annotation_zstack[z]
        labeled_slice = measure.label(mask_slice, connectivity=1)
        new_slice = np.zeros_like(mask_slice)

        for region in measure.regionprops(labeled_slice):
            coords = region.coords
            intensities = raw_slice[coords[:, 0], coords[:, 1]]

            min_intensity = intensities.min()
            max_intensity = intensities.max()

            threshold = min_intensity + (percentile / 100.0) * (max_intensity - min_intensity)

            keep_coords = coords[intensities >= threshold]
            new_slice[keep_coords[:, 0], keep_coords[:, 1]] = 1

        self.annotation_zstack[z] = measure.label(new_slice, connectivity=1)
        self.update_display()
        
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()

    # ----------------------------------------------------------------
    #  Stretch
    # ----------------------------------------------------------------
    def apply_stretch(self):
        if self.raw_display_zstack is None:
            return
        
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        clip_str = self.stretch_edit.text().strip()
        try:
            clip_percent = float(clip_str)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid percentage. e.g. 1.0 or 2.5")
            return
        
        if clip_percent<0 or clip_percent>50:
            QtWidgets.QMessageBox.warning(self, "Error", "Clip percent must be in [0,50].")
            return
        
        Z = self.raw_display_zstack.shape[0]
        self.stretched_display_zstack = np.zeros_like(self.raw_display_zstack)
        
        lower_p = clip_percent
        upper_p = 100.0 - clip_percent
        
        for z in range(Z):
            slc = self.raw_display_zstack[z].astype(np.float32)
            lowv = np.percentile(slc, lower_p)
            highv = np.percentile(slc, upper_p)
            slc_clipped = np.clip(slc, lowv, highv)
            if highv>lowv:
                slc_clipped = (slc_clipped - lowv)/(highv - lowv)*255.0
            self.stretched_display_zstack[z] = slc_clipped.astype(np.uint8)
        
        self.use_stretched = True
        self.update_display()
        
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()
    
    def reset_stretch(self):
        if self.raw_display_zstack is not None:
            xlim = self.canvas.axes.get_xlim()
            ylim = self.canvas.axes.get_ylim()
            
            self.use_stretched = False
            self.update_display()
            
            self.canvas.axes.set_xlim(xlim)
            self.canvas.axes.set_ylim(ylim)
            self.canvas.draw()

    # ----------------------------------------------------------------
    #  Selection & Deletion => can also be undone
    # ----------------------------------------------------------------
    def toggle_select_mode(self):
        self.select_mode = not self.select_mode
        if self.select_mode:
            self.select_mode_button.setText("Exit Select Mode")
            self.statusBar().showMessage("Select Mode ON: click or drag to delete mask(s).")
        else:
            self.select_mode_button.setText("Select Mode")
            self.statusBar().clearMessage()
    
    def _delete_single_area(self, x, y):
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        if self.annotation_zstack is None:
            return
        
        # push undo
        self.push_undo_stack()
        
        z = self.current_z
        current_slice = self.annotation_zstack[z]
        labeled_slice = measure.label(current_slice, connectivity=1, background=0)
        label_val = labeled_slice[y, x]
        if label_val != 0:
            labeled_slice[labeled_slice==label_val] = 0
            self.annotation_zstack[z] = labeled_slice
            self.update_display()
        
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()
    
    def _delete_areas_in_rect(self, xmin, ymin, xmax, ymax):
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        if self.annotation_zstack is None:
            return
        
        # push undo
        self.push_undo_stack()

        z = self.current_z
        current_slice = self.annotation_zstack[z]
        labeled_slice = measure.label(current_slice, connectivity=1, background=0)
        
        H, W = labeled_slice.shape
        xmin = max(0, min(W-1, xmin))
        xmax = max(0, min(W-1, xmax))
        ymin = max(0, min(H-1, ymin))
        ymax = max(0, min(H-1, ymax))
        
        if xmax < xmin or ymax < ymin:
            return
        
        box_labels = np.unique(labeled_slice[ymin:ymax+1, xmin:xmax+1])
        for lv in box_labels:
            if lv!=0:
                labeled_slice[labeled_slice==lv] = 0
        
        self.annotation_zstack[z] = labeled_slice
        self.update_display()
        
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()
    
    # ----------------------------------------------------------------
    #  Mouse events
    # ----------------------------------------------------------------
    def on_mouse_press(self, event):
        if event.inaxes != self.canvas.axes:
            return
        
        # Right-click => pan
        if event.button == 3:
            self._panning = True
            self._pan_start_xy = (event.x, event.y)
            self._pan_start_xlim = self.canvas.axes.get_xlim()
            self._pan_start_ylim = self.canvas.axes.get_ylim()
            return
        
        # Left-click + select_mode => start rectangle
        if event.button == 1 and self.select_mode:
            self._dragging_rect = True
            self._rect_start = (event.xdata, event.ydata)
            
            if self._rect_patch is not None:
                self._rect_patch.remove()
            
            self._rect_patch = Rectangle(
                xy=(self._rect_start[0], self._rect_start[1]),
                width=0,
                height=0,
                linewidth=1.5,
                edgecolor='yellow',
                facecolor='none'
            )
            self.canvas.axes.add_patch(self._rect_patch)
            self.canvas.draw()

    def on_mouse_release(self, event):
        if event.button == 3:
            self._panning = False
            return
        
        if event.button == 1 and self.select_mode and self._dragging_rect:
            self._dragging_rect = False
            if self._rect_patch:
                self._rect_patch.remove()
                self._rect_patch = None
            self.canvas.draw()
            
            x0, y0 = self._rect_start
            x1, y1 = (event.xdata, event.ydata)
            if x1 is None or y1 is None:
                x1, y1 = x0, y0
            
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            
            if dx<2 and dy<2:
                x = int(round(x0))
                y = int(round(y0))
                if x>=0 and y>=0:
                    self._delete_single_area(x, y)
            else:
                xmin = int(round(min(x0,x1)))
                xmax = int(round(max(x0,x1)))
                ymin = int(round(min(y0,y1)))
                ymax = int(round(max(y0,y1)))
                self._delete_areas_in_rect(xmin, ymin, xmax, ymax)

    def on_mouse_move(self, event):
        if self._panning:
            if event.inaxes!=self.canvas.axes:
                return
            if self._pan_start_xlim is None or self._pan_start_ylim is None:
                return
            
            dx = event.x - self._pan_start_xy[0]
            dy = event.y - self._pan_start_xy[1]
            
            ax = self.canvas.axes
            w = ax.bbox.width
            h = ax.bbox.height
            
            xlim0 = self._pan_start_xlim
            ylim0 = self._pan_start_ylim
            
            xr = xlim0[1] - xlim0[0]
            yr = ylim0[1] - ylim0[0]
            
            dx_frac = dx / w
            dy_frac = dy / h
            
            dx_data = dx_frac*xr
            dy_data = dy_frac*yr
            
            new_xlim = (xlim0[0] - dx_data, xlim0[1] - dx_data)
            new_ylim = (ylim0[0] - dy_data, ylim0[1] - dy_data)
            
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            self.canvas.draw()
        
        if self._dragging_rect and event.button == 1 and self.select_mode:
            if event.inaxes!=self.canvas.axes:
                return
            if self._rect_patch:
                x0, y0 = self._rect_start
                x1, y1 = (event.xdata, event.ydata)
                if x1 is None or y1 is None:
                    return
                width = x1 - x0
                height = y1 - y0
                self._rect_patch.set_x(x0)
                self._rect_patch.set_y(y0)
                self._rect_patch.set_width(width)
                self._rect_patch.set_height(height)
                self.canvas.draw()
    
    # ----------------------------------------------------------------
    #  Zoom with mouse wheel
    # ----------------------------------------------------------------
    def on_scroll(self, event):
        if event.inaxes != self.canvas.axes:
            return
        
        # up => zoom in, down => zoom out
        if event.button == 'up':
            scale = 1 / self._zoom_scale
        else:
            scale = self._zoom_scale
        
        xdata = event.xdata
        ydata = event.ydata
        
        cur_xlim = self.canvas.axes.get_xlim()
        cur_ylim = self.canvas.axes.get_ylim()
        
        cur_width = (cur_xlim[1] - cur_xlim[0])
        cur_height = (cur_ylim[1] - cur_ylim[0])
        
        relx = (xdata - cur_xlim[0]) / cur_width if cur_width != 0 else 0
        rely = (ydata - cur_ylim[0]) / cur_height if cur_height != 0 else 0
        
        new_width = cur_width * scale
        new_height = cur_height * scale
        
        self.canvas.axes.set_xlim([
            xdata - relx * new_width,
            xdata + (1 - relx) * new_width
        ])
        self.canvas.axes.set_ylim([
            ydata - rely * new_height,
            ydata + (1 - rely) * new_height
        ])
        
        self.canvas.draw()

    # ----------------------------------------------------------------
    #  Update Masks
    # ----------------------------------------------------------------
    def update_masks(self):
        if self.annotation_zstack is None:
            return
        
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        Z = self.annotation_zstack.shape[0]
        new_anno = np.zeros_like(self.annotation_zstack)
        
        for z in range(Z):
            labeled_slice = measure.label(self.annotation_zstack[z], connectivity=1, background=0)
            new_anno[z] = labeled_slice
        
        self.annotation_zstack = new_anno
        self.update_display()
        
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()
        
        QtWidgets.QMessageBox.information(
            self,
            "Update Masks",
            "All slices have been re-labeled. Morphological operations will now use the new labels."
        )

    # ----------------------------------------------------------------
    #  Save (with sub-stack) => unify all >0 => 1
    # ----------------------------------------------------------------
    def save_annotation(self):
        if self.annotation_zstack is None or self.original_zstack is None:
            QtWidgets.QMessageBox.information(self, "Info", "No data to save.")
            return
        
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        # 先尝试读 start_z / end_z
        Z = self.original_zstack.shape[0]
        try:
            start_z = int(self.substack_start_edit.text())
            end_z = int(self.substack_end_edit.text())
        except ValueError:
            start_z = 0
            end_z = Z - 1
        
        if start_z<0: 
            start_z = 0
        if end_z >= Z:
            end_z = Z - 1
        if end_z < start_z:
            QtWidgets.QMessageBox.warning(self, "Error", f"Invalid sub-stack range: must be 0 <= start <= end < {Z}")
            return
        
        sub_anno = self.annotation_zstack[start_z:end_z+1]
        sub_anno[sub_anno>0] = 1
        
        if self.original_zstack.ndim == 3:
            sub_orig = self.original_zstack[start_z:end_z+1,:,:]
        else:
            sub_orig = self.original_zstack[start_z:end_z+1,:,:,:]
        
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Annotation Sub-stack as TIFF",
            "",
            "TIF Files (*.tif);;All Files (*)"
        )
        if not save_path:
            return
        
        try:
            tifffile.imwrite(save_path, sub_anno.astype(np.uint16))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save annotation: {e}")
            return
        
        base, ext = os.path.splitext(save_path)
        orig_path = base + "_original.tif"
        
        try:
            tifffile.imwrite(orig_path, sub_orig)
            QtWidgets.QMessageBox.information(
                self, 
                "Success", 
                f"Annotation saved (binary): {save_path}\nOriginal sub-stack saved: {orig_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save original sub-stack: {e}")
        
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()

    # ----------------------------------------------------------------
    #  NEW: 随机播种功能 (UI文本已改为top 40%)
    # ----------------------------------------------------------------
    def random_seeds_current_slice(self):
        if self.annotation_zstack is None or self.original_zstack is None:
            QtWidgets.QMessageBox.information(self, "Info", "No data to do random seeds.")
            return
        
        self.push_undo_stack()  # 备份
        
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        z = self.current_z
        # 取当前切片的原图(8bit)
        raw_slice = self._extract_slice(self.original_zstack, z)

        # 依然用 p60 代表 top 40%，也可以改其他分位
        p60 = np.percentile(raw_slice, 60)
        bright_mask = (raw_slice >= p60)
        
        coords = np.column_stack(np.where(bright_mask))
        if coords.shape[0] == 0:
            QtWidgets.QMessageBox.information(self, "Info", "No pixels above 60th percentile in this slice.")
            return
        
        num_points = coords.shape[0]
        choose_num = min(num_points, 2000)
        chosen_indices = np.random.choice(num_points, choose_num, replace=False)
        chosen_coords = coords[chosen_indices]
        
        # 把这些坐标赋值给 annotation_zstack
        for (r, c) in chosen_coords:
            self.annotation_zstack[z, r, c] = 1
        
        # 更新显示
        self.update_display()
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()
        
        QtWidgets.QMessageBox.information(
            self, 
            "Random Seeds",
            f"Selected {choose_num} bright pixels (>=60th percentile) as seeds in Z={z}."
        )

    # ----------------------------------------------------------------
    #  NEW: 基于像素亮度直方图梯度的扩展 + 第二个阈值
    # ----------------------------------------------------------------
    def gradient_expand_current_slice(self):
        """
        对当前切片所有已标注区域（label）做梯度扩展：
          1. 对每个 label 区域，统计该区域在原图中的平均强度 mean_intensity。
          2. 用户输入一个百分比X => difference_threshold = mean_intensity * (X/100)
          3. 从该区域边缘开始，用BFS/DFS对周边像素进行检验：如果未标注且像素强度与 mean_intensity 差值不超过 threshold，则并入区域并继续扩张。
          4. 但要求：像素本身也要高于切片范围中某个阈值（slice_threshold）。
             如果像素强度 < slice_threshold，则跳过，不做扩张。
          5. 对所有 label 做完后，重新 label 一下整张 mask。
        """
        if self.annotation_zstack is None or self.original_zstack is None:
            QtWidgets.QMessageBox.information(self, "Info", "No data to do gradient expansion.")
            return
        
        # 先 push_undo_stack
        self.push_undo_stack()

        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()

        z = self.current_z
        raw_slice = self._extract_slice(self.original_zstack, z)
        annotated_slice = self.annotation_zstack[z]
        
        # 解析用户指定的梯度百分比
        try:
            grad_str = self.gradient_expand_edit.text()
            grad_percent = float(grad_str)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Please input a valid float for gradient expand (%).")
            return
        
        # 解析第二个阈值：只允许在该 slice 的 [min, max] 区间中，超过一定百分比的像素才能被考虑扩张
        try:
            slice_thresh_str = self.gradient_slice_thresh_edit.text()
            slice_percent = float(slice_thresh_str)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Please input a valid float for min slice intensity (%).")
            return
        
        slice_min = float(raw_slice.min())
        slice_max = float(raw_slice.max())
        slice_thresh = slice_min + (slice_percent/100.0) * (slice_max - slice_min)
        
        # 先对已有标注做 measure.label
        labeled = measure.label(annotated_slice, connectivity=1, background=0)
        labels = np.unique(labeled)
        labels = labels[labels != 0]  # 去掉背景0
        
        # 8邻接（上下左右+对角）
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            ( 0, -1),           ( 0, 1),
                            ( 1, -1), ( 1, 0), ( 1, 1)]
        
        H, W = labeled.shape
        
        # 对每个 label 做 region growing
        for lv in labels:
            region_mask = (labeled == lv)
            coords = np.column_stack(np.where(region_mask))
            if coords.shape[0] == 0:
                continue
            
            # 计算区域平均值
            intensities = raw_slice[coords[:,0], coords[:,1]].astype(np.float32)
            mean_intensity = intensities.mean()
            diff_threshold = mean_intensity * (grad_percent / 100.0)
            
            # BFS队列初始化: 把 region 所有像素都作为起点
            queue = list(coords)
            visited = set((r, c) for (r, c) in coords)  # 已经在 region 内 / 已访问
            
            while queue:
                r, c = queue.pop()
                for dr, dc in neighbor_offsets:
                    rr = r + dr
                    cc = c + dc
                    if rr<0 or rr>=H or cc<0 or cc>=W:
                        continue
                    if (rr, cc) in visited:
                        continue
                    # 如果邻居已经属于某个别的 label，就跳过
                    if labeled[rr, cc] != 0:
                        continue
                    
                    neighbor_int = float(raw_slice[rr, cc])
                    # 第二阈值: 如果该像素低于 slice_thresh，不进行扩张
                    if neighbor_int < slice_thresh:
                        continue
                    
                    # 第一步：与mean_intensity 的差值是否 <= diff_threshold
                    if abs(neighbor_int - mean_intensity) <= diff_threshold:
                        labeled[rr, cc] = lv
                        visited.add((rr, cc))
                        queue.append((rr, cc))
        
        # 由于可能出现两个区域互相靠近、合并等，需要再整体 measure.label 一下
        labeled_final = measure.label((labeled>0).astype(np.uint8), connectivity=1, background=0)
        self.annotation_zstack[z] = labeled_final
        
        self.update_display()
        self.canvas.axes.set_xlim(xlim)
        self.canvas.axes.set_ylim(ylim)
        self.canvas.draw()

        QtWidgets.QMessageBox.information(
            self,
            "Gradient Expand",
            "Gradient expansion done for current slice!"
        )


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ZStackAnnotationTool()
    window.show()
    app.exec_()

if __name__=="__main__":
    main()
