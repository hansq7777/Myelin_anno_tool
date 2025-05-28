# Function Reference

This document summarizes key functions in the codebase with a short description of their purpose or main implementation points.

## Summary Table

| Function | Implementation notes / purpose |
|---------|--------------------------------|
| ~~MplCanvas.__init__~~ | ~~calls Figure, __init__, add_subplot, setParent, super, tight_layout~~ |
| ~~ZStackAnnotationTool.__init__~~ | ~~calls MplCanvas, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QVBoxLayout, QWidget, __init__, addLayout~~ |
| ~~ZStackAnnotationTool.keyPressEvent~~ | ~~calls apply_filter_current_slice, dilate_current_slice, erode_current_slice, focusWidget, gradient_expand_current_slice, key, keyPressEvent, next_slice, prev_slice, random_seeds_current_slice~~ |
| ~~ZStackAnnotationTool.push_undo_stack~~ | ~~invoked before changes; saves current state to `undo_stack` and clears `redo_stack`~~ |
| ~~ZStackAnnotationTool.undo_last_operation~~ | ~~pops from `undo_stack`, pushes onto `redo_stack`, and restores it~~ |
| ~~ZStackAnnotationTool.redo_operation~~ | ~~pops from `redo_stack`, pushes onto `undo_stack`, and restores it~~ |
| ~~ZStackAnnotationTool.prev_slice~~ | ~~calls max, setValue~~ |
| ~~ZStackAnnotationTool.next_slice~~ | ~~calls min, setValue~~ |
| ~~ZStackAnnotationTool.closeEvent~~ | ~~calls accept, ignore, warning~~ |
| ZStackAnnotationTool._on_close_window | calls close, instance, quit |
| ZStackAnnotationTool.choose_annotation_folder | calls extend, getExistingDirectory, glob, information, join, sort |
| ~~ZStackAnnotationTool.choose_image_file~~ | ~~calls getOpenFileName~~ |
| ~~ZStackAnnotationTool.choose_annotation_files~~ | ~~calls extend, getOpenFileNames~~ |
| ~~ZStackAnnotationTool.load_data~~ | ~~calls ValueError, _extract_slice, append, astype, clear, copy, imread, isfile, len, logical_or~~ |
| ZStackAnnotationTool.close_current | calls clear, draw, setEnabled, setText, setValue |
| ~~ZStackAnnotationTool.on_slice_changed~~ | ~~calls draw, get_xlim, get_ylim, set_xlim, set_ylim, update_display~~ |
| ~~ZStackAnnotationTool.update_display~~ | ~~calls astype, clear, draw, imshow, label, len, regionprops, setText, sum, zeros~~ |
| ZStackAnnotationTool._extract_slice | calls ValueError, _normalize_to_8bit, mean |
| ZStackAnnotationTool._normalize_to_8bit | calls array, astype, max, min |
| ~~ZStackAnnotationTool.apply_filter_current_slice~~ | ~~calls draw, get_xlim, get_ylim, int, label, push_undo_stack, remove_small_objects, set_xlim, set_ylim, text~~ |
| ~~ZStackAnnotationTool.erode_current_slice~~ | ~~calls binary_erosion, diamond, draw, get_xlim, get_ylim, int, label, push_undo_stack, set_xlim, set_ylim~~ |
| ~~ZStackAnnotationTool.dilate_current_slice~~ | ~~calls binary_dilation, diamond, draw, get_xlim, get_ylim, int, label, push_undo_stack, set_xlim, set_ylim~~ |
| ~~ZStackAnnotationTool.remove_bg_pixels_current_slice~~ | ~~calls _extract_slice, draw, float, get_xlim, get_ylim, label, max, min, push_undo_stack, regionprops~~ |
| ~~ZStackAnnotationTool.apply_stretch~~ | ~~calls astype, clip, draw, float, get_xlim, get_ylim, percentile, range, set_xlim, set_ylim~~ |
| ~~ZStackAnnotationTool.reset_stretch~~ | ~~calls draw, get_xlim, get_ylim, set_xlim, set_ylim, update_display~~ |
| ZStackAnnotationTool.toggle_select_mode | calls clearMessage, setText, showMessage, statusBar |
| ZStackAnnotationTool._delete_single_area | calls draw, get_xlim, get_ylim, label, push_undo_stack, set_xlim, set_ylim, update_display |
| ~~ZStackAnnotationTool._delete_areas_in_rect~~ | ~~calls draw, get_xlim, get_ylim, label, max, min, push_undo_stack, set_xlim, set_ylim, unique~~ |
| ~~ZStackAnnotationTool.on_mouse_press~~ | ~~calls Rectangle, add_patch, draw, get_xlim, get_ylim, remove~~ |
| ~~ZStackAnnotationTool.on_mouse_release~~ | ~~calls _delete_areas_in_rect, _delete_single_area, abs, draw, int, max, min, remove, round~~ |
| ~~ZStackAnnotationTool.on_mouse_move~~ | ~~calls draw, set_height, set_width, set_x, set_xlim, set_y, set_ylim~~ |
| ZStackAnnotationTool.on_scroll | calls draw, get_xlim, get_ylim, set_xlim, set_ylim |
| ZStackAnnotationTool.update_masks | calls draw, get_xlim, get_ylim, information, label, range, set_xlim, set_ylim, update_display, zeros_like |
| ~~ZStackAnnotationTool.save_annotation~~ | ~~calls astype, draw, getSaveFileName, get_xlim, get_ylim, imwrite, information, int, set_xlim, set_ylim~~ |
| ~~ZStackAnnotationTool.random_seeds_current_slice~~ | ~~calls _extract_slice, choice, column_stack, draw, get_xlim, get_ylim, information, min, percentile, push_undo_stack~~ |
| ~~ZStackAnnotationTool.gradient_expand_current_slice~~ | ~~expands labels based on intensity differences using BFS/DFS with a slice threshold; relabels mask afterward~~ |
| ~~main~~ | ~~calls QApplication, ZStackAnnotationTool, exec_, show~~ |

## Checklist

- [x] MplCanvas.__init__
- [x] ZStackAnnotationTool.__init__
- [x] ZStackAnnotationTool.keyPressEvent
- [x] ZStackAnnotationTool.push_undo_stack
- [x] ZStackAnnotationTool.undo_last_operation
- [x] ZStackAnnotationTool.redo_operation
- [x] ZStackAnnotationTool.prev_slice
- [x] ZStackAnnotationTool.next_slice
- [x] ZStackAnnotationTool.closeEvent
- [ ] ZStackAnnotationTool._on_close_window
- [ ] ZStackAnnotationTool.choose_annotation_folder
- [x] ZStackAnnotationTool.choose_image_file
- [x] ZStackAnnotationTool.choose_annotation_files
- [x] ZStackAnnotationTool.load_data
- [ ] ZStackAnnotationTool.close_current
- [x] ZStackAnnotationTool.on_slice_changed
- [x] ZStackAnnotationTool.update_display
- [ ] ZStackAnnotationTool._extract_slice
- [ ] ZStackAnnotationTool._normalize_to_8bit
- [x] ZStackAnnotationTool.apply_filter_current_slice
- [x] ZStackAnnotationTool.erode_current_slice
- [x] ZStackAnnotationTool.dilate_current_slice
- [x] ZStackAnnotationTool.remove_bg_pixels_current_slice
- [x] ZStackAnnotationTool.apply_stretch
- [x] ZStackAnnotationTool.reset_stretch
- [ ] ZStackAnnotationTool.toggle_select_mode
- [ ] ZStackAnnotationTool._delete_single_area
- [x] ZStackAnnotationTool._delete_areas_in_rect
- [x] ZStackAnnotationTool.on_mouse_press
- [x] ZStackAnnotationTool.on_mouse_release
- [x] ZStackAnnotationTool.on_mouse_move
- [ ] ZStackAnnotationTool.on_scroll
- [ ] ZStackAnnotationTool.update_masks
- [x] ZStackAnnotationTool.save_annotation
- [x] ZStackAnnotationTool.random_seeds_current_slice
- [x] ZStackAnnotationTool.gradient_expand_current_slice
- [x] main

