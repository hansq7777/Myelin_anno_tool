from __future__ import annotations

import numpy as np
import threading
import time
from typing import TYPE_CHECKING, Callable, Sequence

from PyQt5.QtCore import QThread, pyqtSignal

from ..utils import morphology_tools

if TYPE_CHECKING:  # pragma: no cover
    from .main_controller import MainController


class QuickAutoCancelled(RuntimeError):
    """Raised when a background quick auto run is cancelled."""


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise QuickAutoCancelled("Quick auto cancelled")


def _quick_auto_slice_pipeline_for_arrays(
    data_stack: np.ndarray,
    mask_stack: np.ndarray | None,
    slice_index: int,
    original_mask: np.ndarray,
    *,
    seed_percentile: float = 85.0,
    seed_pixel_percent: float = 1.0,
    dilate_iterations: int = 1,
    bg1_percentile: float = 8.0,
    bg1_bins: int = 0,
    grow_diff_pct: float = 35.0,
    grow_hist_pct: float = 20.0,
    grow_force_pct: float = -1.0,
    grow_limit: int = 30000,
    bg2_percentile: float = 12.0,
    bg2_bins: int = 0,
    final_bg_repeat: int = 5,
    final_bg_percentile: float = 5.0,
    final_bg_bins: int = 5,
    final_small_threshold: int = 20,
    addition_support_percentile: float = 75.0,
    protect_small_original: bool = True,
    small_component_guard: int = 120,
    cancel_event: threading.Event | None = None,
) -> np.ndarray:
    img = np.asarray(data_stack[slice_index])
    cur_mask = np.asarray(original_mask, dtype=np.uint8).copy()
    other_fg_values = np.array([], dtype=img.dtype)
    if mask_stack is not None:
        other_masks = np.asarray(mask_stack).copy()
        other_masks[slice_index] = 0
        other_fg_values = np.asarray(data_stack)[other_masks > 0]

    def global_threshold(bins: int, slice_mask: np.ndarray) -> float | None:
        if bins <= 0:
            return None
        current_values = img[slice_mask > 0]
        if other_fg_values.size > 0 and current_values.size > 0:
            values = np.concatenate((other_fg_values, current_values))
        elif other_fg_values.size > 0:
            values = other_fg_values
        else:
            values = current_values
        if values.size == 0:
            return None
        _hist, edges = np.histogram(values, bins=256)
        idx = min(int(bins), len(edges) - 2)
        return float(edges[idx])

    _raise_if_cancelled(cancel_event)
    seeds = morphology_tools.sample_seeds(
        img,
        seed_percentile,
        pixel_percent=seed_pixel_percent,
    )
    cur_mask[seeds > 0] = 1
    _raise_if_cancelled(cancel_event)
    cur_mask = morphology_tools.dilate(cur_mask, iterations=dilate_iterations)
    _raise_if_cancelled(cancel_event)
    cur_mask = morphology_tools.remove_mask_background(
        img,
        cur_mask,
        bg1_percentile,
        global_threshold(bg1_bins, cur_mask),
    )
    _raise_if_cancelled(cancel_event)

    hist_pct = grow_hist_pct if grow_hist_pct >= 0 else None
    force_pct = grow_force_pct if grow_force_pct >= 0 else None
    limit = grow_limit if grow_limit > 0 else None
    cur_mask = morphology_tools.intensity_region_grow(
        img.astype(float),
        cur_mask,
        grow_diff_pct,
        hist_pct,
        force_pct,
        limit,
        cancel_event=cancel_event,
    )

    _raise_if_cancelled(cancel_event)
    cur_mask = morphology_tools.remove_mask_background(
        img,
        cur_mask,
        bg2_percentile,
        global_threshold(bg2_bins, cur_mask),
    )
    for _ in range(max(0, int(final_bg_repeat))):
        _raise_if_cancelled(cancel_event)
        cur_mask = morphology_tools.remove_mask_background(
            img,
            cur_mask,
            final_bg_percentile,
            global_threshold(final_bg_bins, cur_mask),
        )

    if final_small_threshold > 0 and mask_stack is not None:
        _raise_if_cancelled(cancel_event)
        temp_stack = np.asarray(mask_stack).copy()
        temp_stack[slice_index] = cur_mask
        filtered_stack, _labels = morphology_tools.remove_small_components(
            temp_stack,
            int(final_small_threshold),
        )
        cur_mask = filtered_stack[slice_index]

    if addition_support_percentile >= 0:
        _raise_if_cancelled(cancel_event)
        img_float = img.astype(float)
        support_thresh = float(np.percentile(img_float, addition_support_percentile))
        support = img_float >= support_thresh
        additions = (cur_mask > 0) & (original_mask == 0)
        cur_mask[additions & ~support] = 0

    if protect_small_original and small_component_guard > 0:
        _raise_if_cancelled(cancel_event)
        labels = morphology_tools.label_components(original_mask)
        if labels.size > 0:
            counts = np.bincount(labels.ravel())
            if counts.size > 1:
                keep_ids = np.where(
                    (np.arange(counts.size) > 0) & (counts <= small_component_guard)
                )[0]
                if keep_ids.size > 0:
                    protected = np.isin(labels, keep_ids)
                    cur_mask[protected] = 1

    _raise_if_cancelled(cancel_event)
    return cur_mask.astype(np.uint8, copy=False)


def _run_quick_auto_single(
    data_stack: np.ndarray,
    mask_stack: np.ndarray,
    slice_index: int,
    *,
    cancel_event: threading.Event | None = None,
    **params: object,
) -> dict[str, object]:
    original_mask = np.asarray(mask_stack[slice_index], dtype=np.uint8).copy()
    before_pixels = int(np.count_nonzero(original_mask))
    mask = _quick_auto_slice_pipeline_for_arrays(
        data_stack,
        mask_stack,
        slice_index,
        original_mask,
        cancel_event=cancel_event,
        **params,
    )
    after_pixels = int(np.count_nonzero(mask))
    return {
        "slice_index": int(slice_index),
        "mask": mask,
        "before_pixels": before_pixels,
        "after_pixels": after_pixels,
        "changed": not np.array_equal(mask, original_mask),
    }


def _run_quick_auto_stack(
    data_stack: np.ndarray,
    mask_stack: np.ndarray,
    indices: Sequence[int],
    *,
    cancel_event: threading.Event | None = None,
    progress_fn: Callable[[int, int, int, dict[str, object]], None] | None = None,
    **params: object,
) -> dict[str, object]:
    work_masks = np.asarray(mask_stack, dtype=np.uint8).copy()
    ordered_indices = [int(idx) for idx in indices]
    metrics_by_slice: list[dict[str, object]] = []
    total = len(ordered_indices)
    for processed, slice_index in enumerate(ordered_indices, start=1):
        _raise_if_cancelled(cancel_event)
        metrics = _run_quick_auto_single(
            data_stack,
            work_masks,
            slice_index,
            cancel_event=cancel_event,
            **params,
        )
        work_masks[slice_index] = np.asarray(metrics["mask"], dtype=np.uint8)
        metrics_by_slice.append(metrics)
        if progress_fn is not None:
            progress_fn(processed, total, slice_index, metrics)
    _raise_if_cancelled(cancel_event)
    return {
        "masks": work_masks,
        "indices": ordered_indices,
        "metrics_by_slice": metrics_by_slice,
        "changed": any(bool(m.get("changed")) for m in metrics_by_slice),
    }


class QuickAutoThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(object)
    cancelled = pyqtSignal(object)
    progress = pyqtSignal(object)

    def __init__(
        self,
        *,
        data_stack: np.ndarray,
        mask_stack: np.ndarray,
        params: dict[str, object],
        request_id: int,
        image_revision: int,
        mask_revision: int,
        mode: str,
        indices: Sequence[int],
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.data_stack = np.asarray(data_stack)
        self.mask_stack = np.asarray(mask_stack, dtype=np.uint8)
        self.params = dict(params)
        self.request_id = int(request_id)
        self.image_revision = int(image_revision)
        self.mask_revision = int(mask_revision)
        self.mode = str(mode)
        self.indices = [int(idx) for idx in indices]
        self.cancel_event = cancel_event

    def run(self) -> None:
        started = time.monotonic()
        processed = 0
        total = len(self.indices)

        def progress_cb(
            cur: int,
            total_count: int,
            slice_index: int,
            metrics: dict[str, object],
        ) -> None:
            nonlocal processed
            processed = int(cur)
            self.progress.emit(
                {
                    "request_id": self.request_id,
                    "mode": self.mode,
                    "processed": int(cur),
                    "total": int(total_count),
                    "slice_index": int(slice_index),
                    "metrics": metrics,
                }
            )

        try:
            if self.mode == "single":
                if not self.indices:
                    raise ValueError("Quick auto single mode requires one slice index")
                result = _run_quick_auto_single(
                    self.data_stack,
                    self.mask_stack,
                    self.indices[0],
                    cancel_event=self.cancel_event,
                    **self.params,
                )
                processed = 1
            elif self.mode == "stack":
                result = _run_quick_auto_stack(
                    self.data_stack,
                    self.mask_stack,
                    self.indices,
                    cancel_event=self.cancel_event,
                    progress_fn=progress_cb,
                    **self.params,
                )
                processed = total
            else:
                raise ValueError(f"Unsupported quick auto mode: {self.mode}")
        except QuickAutoCancelled:
            self.cancelled.emit(
                {
                    "request_id": self.request_id,
                    "mode": self.mode,
                    "processed": processed,
                    "total": total,
                    "elapsed_sec": time.monotonic() - started,
                }
            )
            return
        except Exception as exc:
            self.failed.emit(
                {
                    "request_id": self.request_id,
                    "mode": self.mode,
                    "processed": processed,
                    "total": total,
                    "elapsed_sec": time.monotonic() - started,
                    "error": str(exc),
                }
            )
            return
        self.succeeded.emit(
            {
                "request_id": self.request_id,
                "mode": self.mode,
                "image_revision": self.image_revision,
                "mask_revision": self.mask_revision,
                "processed": processed,
                "total": total,
                "elapsed_sec": time.monotonic() - started,
                **result,
            }
        )


class ScriptMixin:
    """Methods used by the ScriptEditor."""

    def _quick_auto_slice_pipeline(
        self: "MainController",
        original_mask: np.ndarray,
        *,
        seed_percentile: float,
        seed_pixel_percent: float,
        dilate_iterations: int,
        bg1_percentile: float,
        bg1_bins: int,
        grow_diff_pct: float,
        grow_hist_pct: float,
        grow_force_pct: float,
        grow_limit: int,
        bg2_percentile: float,
        bg2_bins: int,
        final_bg_repeat: int,
        final_bg_percentile: float,
        final_bg_bins: int,
        final_small_threshold: int,
        addition_support_percentile: float,
        protect_small_original: bool,
        small_component_guard: int,
        cancel_event: threading.Event | None = None,
    ) -> np.ndarray:
        if self.model.data is None:
            raise RuntimeError("Image stack must be loaded before running quick auto")
        return _quick_auto_slice_pipeline_for_arrays(
            self.model.data,
            self.model.masks,
            self.model.index,
            original_mask,
            seed_percentile=seed_percentile,
            seed_pixel_percent=seed_pixel_percent,
            dilate_iterations=dilate_iterations,
            bg1_percentile=bg1_percentile,
            bg1_bins=bg1_bins,
            grow_diff_pct=grow_diff_pct,
            grow_hist_pct=grow_hist_pct,
            grow_force_pct=grow_force_pct,
            grow_limit=grow_limit,
            bg2_percentile=bg2_percentile,
            bg2_bins=bg2_bins,
            final_bg_repeat=final_bg_repeat,
            final_bg_percentile=final_bg_percentile,
            final_bg_bins=final_bg_bins,
            final_small_threshold=final_small_threshold,
            addition_support_percentile=addition_support_percentile,
            protect_small_original=protect_small_original,
            small_component_guard=small_component_guard,
            cancel_event=cancel_event,
        )

    def script_quick_seed_dilate_bg_int_bg(
        self: "MainController",
        seed_percentile: float = 85.0,
        seed_pixel_percent: float = 1.0,
        dilate_iterations: int = 1,
        bg1_percentile: float = 8.0,
        bg1_bins: int = 0,
        grow_diff_pct: float = 35.0,
        grow_hist_pct: float = 20.0,
        grow_force_pct: float = -1.0,
        grow_limit: int = 30000,
        bg2_percentile: float = 12.0,
        bg2_bins: int = 0,
        final_bg_repeat: int = 5,
        final_bg_percentile: float = 5.0,
        final_bg_bins: int = 5,
        final_small_threshold: int = 20,
        addition_support_percentile: float = 75.0,
        protect_small_original: bool = True,
        small_component_guard: int = 120,
        show_status: bool = True,
        push_undo: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, int | bool] | None:
        """Run the default quick-fix pipeline on the current slice.

        Order:
        1) seed
        2) dilate
        3) background filter
        4) intensity grow
        5) background filter

        Parameters use conservative defaults and are intended to be iterated
        based on user feedback on different datasets.
        """
        if not self._ensure_masks():
            return None

        original_mask = self.model.get_mask().copy()
        before_pixels = int(np.count_nonzero(original_mask))
        start = time.monotonic()
        if push_undo:
            self._push_undo("quick_auto")
        try:
            cur_mask = self._quick_auto_slice_pipeline(
                original_mask,
                seed_percentile=seed_percentile,
                seed_pixel_percent=seed_pixel_percent,
                dilate_iterations=dilate_iterations,
                bg1_percentile=bg1_percentile,
                bg1_bins=bg1_bins,
                grow_diff_pct=grow_diff_pct,
                grow_hist_pct=grow_hist_pct,
                grow_force_pct=grow_force_pct,
                grow_limit=grow_limit,
                bg2_percentile=bg2_percentile,
                bg2_bins=bg2_bins,
                final_bg_repeat=final_bg_repeat,
                final_bg_percentile=final_bg_percentile,
                final_bg_bins=final_bg_bins,
                final_small_threshold=final_small_threshold,
                addition_support_percentile=addition_support_percentile,
                protect_small_original=protect_small_original,
                small_component_guard=small_component_guard,
                cancel_event=cancel_event,
            )
        except QuickAutoCancelled:
            if push_undo:
                self._discard_last_undo()
            return None
        changed = not np.array_equal(cur_mask, original_mask)
        if push_undo and not changed:
            self._discard_last_undo()
        self.model.set_mask(cur_mask)
        self._update_view()
        after_pixels = int(np.count_nonzero(cur_mask))

        elapsed = time.monotonic() - start
        if show_status:
            self.statusBar().showMessage(
                (
                    f"Quick script finished in {elapsed:.2f}s "
                    f"(seed->dilate->bg->grow->bg + tail cleanup) | "
                    f"pixels {before_pixels} -> {after_pixels}"
                )
            )
        return {
            "before_pixels": before_pixels,
            "after_pixels": after_pixels,
            "changed": changed,
        }

    def script_dilate(self: "MainController", iterations: int = 1) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("dilate")
        cur = self.model.get_mask()
        new = morphology_tools.dilate(cur, iterations=iterations)
        self.model.set_mask(new)
        self._update_view()

    def script_erode(self: "MainController", iterations: int = 1) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("erode")
        cur = self.model.get_mask()
        new = morphology_tools.erode(cur, iterations=iterations)
        self.model.set_mask(new)
        self._update_view()

    def script_close(self: "MainController", strength: int = 1) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("close")
        cur = self.model.get_mask()
        new = morphology_tools.close(cur, strength)
        self.model.set_mask(new)
        self._update_view()

    def script_filter_small(self: "MainController", threshold: int = 100) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("filter")
        if self.model.masks is None:
            return
        if threshold <= 1:
            self._discard_last_undo()
            return
        new, labels = morphology_tools.remove_small_components(self.model.masks, threshold)
        self.model.replace_masks(new, components=labels, dirty=True)
        self._update_view()

    def script_skeletonize(
        self: "MainController",
        algorithm: str = "skeletonize",
        return_distance: bool = False,
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
            self.model.replace_masks(result, dirty=True)
        else:
            cur = self.model.get_mask()
            params = {}
            if algorithm == "medial_axis":
                params["return_distance"] = return_distance
            new = morphology_tools.skeletonize_slice(cur, algorithm=algorithm, **params)
            self.model.set_mask(new)
        self._update_view()

    def script_threshold_abs(self: "MainController", value: float = 0.0) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("thresh_abs")
        self.model.threshold_absolute(value)
        self._update_view()

    def script_threshold_norm(self: "MainController", percent: float = 50.0) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("thresh_norm")
        self.model.threshold_normalized(percent)
        self._update_view()

    def script_seed(
        self: "MainController", percentile: float = 90.0, pixel_percent: float = 1.0
    ) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("seed")
        img = self.model._extract_slice(self.model.index)
        cur = self.model.get_mask()
        seeds = morphology_tools.sample_seeds(
            img, percentile, pixel_percent=pixel_percent
        )
        cur = cur.copy()
        cur[seeds > 0] = 1
        self.model.set_mask(cur)
        self._update_view()

    def script_int_grow(
        self: "MainController",
        diff_pct: float = 20.0,
        hist_pct: float | None = None,
        force_pct: float | None = None,
        limit: int | None = 30000,
    ) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("int_grow")
        img = self.model._extract_slice(self.model.index)
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
        self: "MainController",
        connectivity: int = 1,
        tolerance: float = 5.0,
        workers: int = 1,
    ) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("flood_grow")
        img = self.model._extract_slice(self.model.index)
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

    def script_blur(self: "MainController", sigma: float = 1.0) -> None:
        if self.model.data is None:
            return
        self.model.apply_gaussian_blur(sigma)
        self._update_view()

    def script_clear_blur(self: "MainController") -> None:
        self.model.remove_gaussian_blur()
        self._update_view()

    def script_reverse_image(self: "MainController") -> None:
        if self.model.data is None:
            return
        self.model.toggle_reverse_intensity()
        self._update_view()

    def script_frangi_filter(
        self: "MainController",
        sigma_start: float = 1.0,
        sigma_end: float = 3.0,
        sigma_step: float = 1.0,
        threshold: float = 0.5,
        black_ridges: bool = True,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        sigmas = np.arange(sigma_start, sigma_end + sigma_step, sigma_step)
        response = morphology_tools.frangi_filter_slice(
            img, sigmas=sigmas, black_ridges=black_ridges
        )
        mask = (response > threshold).astype(np.uint8)
        self._push_undo("frangi")
        self.model.set_mask(mask)
        self._update_view()

    def script_sato_filter(
        self: "MainController",
        sigma_start: float = 1.0,
        sigma_end: float = 3.0,
        sigma_step: float = 1.0,
        threshold: float = 0.5,
        black_ridges: bool = True,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        sigmas = np.arange(sigma_start, sigma_end + sigma_step, sigma_step)
        response = morphology_tools.sato_filter_slice(
            img, sigmas=sigmas, black_ridges=black_ridges
        )
        mask = (response > threshold).astype(np.uint8)
        self._push_undo("sato")
        self.model.set_mask(mask)
        self._update_view()

    def script_meijering_filter(
        self: "MainController",
        sigma_start: float = 1.0,
        sigma_end: float = 3.0,
        sigma_step: float = 1.0,
        threshold: float = 0.5,
        black_ridges: bool = True,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        sigmas = np.arange(sigma_start, sigma_end + sigma_step, sigma_step)
        response = morphology_tools.meijering_filter_slice(
            img, sigmas=sigmas, black_ridges=black_ridges
        )
        mask = (response > threshold).astype(np.uint8)
        self._push_undo("meijering")
        self.model.set_mask(mask)
        self._update_view()

    def script_shortest_path(
        self: "MainController",
        y0: int = 0,
        x0: int = 0,
        y1: int = 10,
        x1: int = 10,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        path = morphology_tools.shortest_path_slice(img, (y0, x0), (y1, x1))
        cur = self.model.get_mask()
        cur[path > 0] = 1
        self._push_undo("shortest_path")
        self.model.set_mask(cur)
        self._update_view()

    def script_felzenszwalb(
        self: "MainController",
        scale: float = 100.0,
        sigma: float = 0.8,
        min_size: int = 20,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        labels = morphology_tools.felzenszwalb_slice(
            img, scale=scale, sigma=sigma, min_size=min_size
        )
        mask = (labels > labels.min()).astype(np.uint8)
        self._push_undo("felzenszwalb")
        self.model.set_mask(mask)
        self._update_view()

    def script_watershed_ift(self: "MainController") -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        markers = morphology_tools.label_components(self.model.get_mask())
        labels = morphology_tools.watershed_ift_slice(img, markers)
        mask = (labels > 0).astype(np.uint8)
        self._push_undo("watershed_ift")
        self.model.set_mask(mask)
        self._update_view()

    def script_scikit_fmm(
        self: "MainController", max_distance: float = 10.0
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        seeds = self.model.get_mask()
        dist = morphology_tools.fmm_distance_slice(img, seeds)
        mask = (dist <= max_distance).astype(np.uint8)
        self._push_undo("scikit_fmm")
        self.model.set_mask(mask)
        self._update_view()

    def script_fast_marching(
        self: "MainController", stopping_value: float = 10.0
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        seeds = self.model.get_mask()
        dist = morphology_tools.sitk_fast_marching_slice(
            img, seeds, stopping_value=stopping_value
        )
        mask = (dist <= stopping_value).astype(np.uint8)
        self._push_undo("fast_marching")
        self.model.set_mask(mask)
        self._update_view()

    def script_opencv_ridge(
        self: "MainController", threshold: float = 0.5
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        response = morphology_tools.opencv_ridge_filter_slice(img)
        mask = (response > threshold).astype(np.uint8)
        self._push_undo("opencv_ridge")
        self.model.set_mask(mask)
        self._update_view()

    def script_steger_ridge(
        self: "MainController", sigma: float = 1.0, threshold: float = 0.5
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        response = morphology_tools.steger_ridge_filter_slice(img, sigma=sigma)
        mask = (response > threshold).astype(np.uint8)
        self._push_undo("steger_ridge")
        self.model.set_mask(mask)
        self._update_view()

    def script_chan_vese(
        self: "MainController",
        iterations: int = 50,
        smoothing: int = 1,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        mask = morphology_tools.chan_vese_slice(
            img,
            iterations=iterations,
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        self._push_undo("chan_vese")
        self.model.set_mask(mask)
        self._update_view()

    def script_ced_filter(
        self: "MainController", iterations: int = 5
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        result = morphology_tools.ced_filter_slice(img, iterations=iterations)
        mask = (result > result.mean()).astype(np.uint8)
        self._push_undo("ced_filter")
        self.model.set_mask(mask)
        self._update_view()

    def script_tubetk_segment(self: "MainController") -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        mask = morphology_tools.tubetk_segment_tubes_slice(img)
        self._push_undo("tubetk_segment")
        self.model.set_mask(mask)
        self._update_view()

    def script_tubetk_seed_path(self: "MainController") -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        seeds = self.model.get_mask()
        mask = morphology_tools.tubetk_seeded_path_slice(img, seeds)
        self._push_undo("tubetk_seed_path")
        self.model.set_mask(mask)
        self._update_view()

    def script_hessian_filter(
        self: "MainController",
        sigma_start: float = 1.0,
        sigma_end: float = 3.0,
        sigma_step: float = 1.0,
        threshold: float = 0.5,
        black_ridges: bool = True,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        sigmas = np.arange(sigma_start, sigma_end + sigma_step, sigma_step)
        resp = morphology_tools.hessian_filter_slice(
            img, sigmas=sigmas, black_ridges=black_ridges
        )
        mask = (resp > threshold).astype(np.uint8)
        self._push_undo("hessian")
        self.model.set_mask(mask)
        self._update_view()

    def script_gabor_filter(
        self: "MainController", frequency: float = 0.1, theta: float = 0.0
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        resp = morphology_tools.gabor_filter_slice(img, frequency=frequency, theta=theta)
        mask = (resp > resp.mean()).astype(np.uint8)
        self._push_undo("gabor")
        self.model.set_mask(mask)
        self._update_view()

    def script_cv_gabor_filter(
        self: "MainController",
        ksize: int = 21,
        sigma: float = 5.0,
        theta: float = 0.0,
        lambd: float = 10.0,
        gamma: float = 0.5,
        psi: float = 0.0,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        resp = morphology_tools.cv_gabor_filter_slice(
            img, ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi
        )
        mask = (resp > resp.mean()).astype(np.uint8)
        self._push_undo("cv_gabor")
        self.model.set_mask(mask)
        self._update_view()

    def script_structure_tensor(
        self: "MainController", sigma: float = 1.0, threshold: float = 0.5
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        resp = morphology_tools.structure_tensor_eigen_slice(img, sigma=sigma)
        mask = (resp > threshold).astype(np.uint8)
        self._push_undo("structure_tensor")
        self.model.set_mask(mask)
        self._update_view()

    def script_stretch(self: "MainController", percentile: float = 0.0) -> None:
        if self.model.data is None:
            return
        if percentile <= 0:
            self.model.reset_contrast()
        else:
            self.model.histogram_stretch(percentile)
        self._update_view()

    def script_bg_filter(
        self: "MainController", percentile: float = 0.0, bins: int = 0
    ) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("bg_filter")
        self._next_progress = 0.2
        self.model.remove_background(
            percentile, bins, progress=True, progress_fn=self._progress_update
        )
        self._update_view()

    def script_check_segment(
        self: "MainController", percentile: float = 5.0, continuous: bool = True
    ) -> bool:
        """Return False to skip segmentation of the current slice."""
        if self.model.data is None:
            return False
        mask = self.model.get_segment_mask(percentile, continuous)
        return bool(mask[self.model.index])

    def script_next_slice(self: "MainController") -> None:
        self._next_slice()

    def script_prev_slice(self: "MainController") -> None:
        self._prev_slice()

    def script_save(self: "MainController") -> None:
        if self.model.masks is None:
            return
        if self.model.mask_path is None:
            self._save_masks()
        else:
            self.model.save_slice()
