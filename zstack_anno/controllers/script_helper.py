from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ..utils import morphology_tools

if TYPE_CHECKING:  # pragma: no cover
    from .main_controller import MainController


class ScriptMixin:
    """Methods used by the ScriptEditor."""

    def script_dilate(self: 'MainController', iterations: int = 1) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("dilate")
        cur = self.model.get_mask()
        new = morphology_tools.dilate(cur, iterations=iterations)
        self.model.set_mask(new)
        self._update_view()

    def script_erode(self: 'MainController', iterations: int = 1) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("erode")
        cur = self.model.get_mask()
        new = morphology_tools.erode(cur, iterations=iterations)
        self.model.set_mask(new)
        self._update_view()

    def script_filter_small(self: 'MainController', threshold: int = 100) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("filter")
        cur = self.model.get_mask()
        new = morphology_tools.remove_small(cur, threshold)
        self.model.set_mask(new)
        self._update_view()

    def script_skeletonize(
        self: 'MainController', algorithm: str = "skeletonize", return_distance: bool = False
    ) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("skeletonize")
        if algorithm == "skeletonize_3d":
            if self.model.masks is None:
                return
            result = morphology_tools.skeletonize_stack(self.model.masks, algorithm=algorithm)
            self.model.masks = result
        else:
            cur = self.model.get_mask()
            params = {}
            if algorithm == "medial_axis":
                params["return_distance"] = return_distance
            new = morphology_tools.skeletonize_slice(cur, algorithm=algorithm, **params)
            self.model.set_mask(new)
        self._update_view()

    def script_threshold_abs(self: 'MainController', value: float = 0.0) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("thresh_abs")
        self.model.threshold_absolute(value)
        self._update_view()

    def script_threshold_norm(self: 'MainController', percent: float = 50.0) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("thresh_norm")
        self.model.threshold_normalized(percent)
        self._update_view()

    def script_seed(self: 'MainController', percentile: float = 90.0) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("seed")
        img = self.model._extract_slice(self.model.index)
        cur = self.model.get_mask()
        seeds = morphology_tools.sample_seeds(img, percentile, num_seeds=20000)
        cur = cur.copy()
        cur[seeds > 0] = 1
        self.model.set_mask(cur)
        self._update_view()

    def script_int_grow(
        self: 'MainController',
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
        self: 'MainController',
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

    def script_blur(self: 'MainController', sigma: float = 1.0) -> None:
        if self.model.data is None:
            return
        self.model.apply_gaussian_blur(sigma)
        self._update_view()

    def script_clear_blur(self: 'MainController') -> None:
        self.model.remove_gaussian_blur()
        self._update_view()

    def script_frangi_filter(
        self: 'MainController',
        sigma_start: float = 1.0,
        sigma_end: float = 3.0,
        sigma_step: float = 1.0,
        threshold: float = 0.5,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        sigmas = np.arange(sigma_start, sigma_end + sigma_step, sigma_step)
        response = morphology_tools.frangi_filter_slice(img, sigmas=sigmas)
        mask = (response > threshold).astype(np.uint8)
        self._push_undo("frangi")
        self.model.set_mask(mask)
        self._update_view()

    def script_sato_filter(
        self: 'MainController',
        sigma_start: float = 1.0,
        sigma_end: float = 3.0,
        sigma_step: float = 1.0,
        threshold: float = 0.5,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        sigmas = np.arange(sigma_start, sigma_end + sigma_step, sigma_step)
        response = morphology_tools.sato_filter_slice(img, sigmas=sigmas)
        mask = (response > threshold).astype(np.uint8)
        self._push_undo("sato")
        self.model.set_mask(mask)
        self._update_view()

    def script_meijering_filter(
        self: 'MainController',
        sigma_start: float = 1.0,
        sigma_end: float = 3.0,
        sigma_step: float = 1.0,
        threshold: float = 0.5,
    ) -> None:
        if not self._ensure_masks():
            return
        img = self.model._extract_slice(self.model.index)
        sigmas = np.arange(sigma_start, sigma_end + sigma_step, sigma_step)
        response = morphology_tools.meijering_filter_slice(img, sigmas=sigmas)
        mask = (response > threshold).astype(np.uint8)
        self._push_undo("meijering")
        self.model.set_mask(mask)
        self._update_view()

    def script_shortest_path(
        self: 'MainController',
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

    def script_stretch(self: 'MainController', percentile: float = 0.0) -> None:
        if self.model.data is None:
            return
        if percentile <= 0:
            self.model.reset_contrast()
        else:
            self.model.histogram_stretch(percentile)
        self._update_view()

    def script_bg_filter(self: 'MainController', percentile: float = 0.0, bins: int = 0) -> None:
        if not self._ensure_masks():
            return
        self._push_undo("bg_filter")
        self._next_progress = 0.2
        self.model.remove_background(percentile, bins, progress=True, progress_fn=self._progress_update)
        self._update_view()

    def script_check_segment(
        self: 'MainController', percentile: float = 5.0, continuous: bool = True
    ) -> bool:
        """Return False to skip segmentation of the current slice."""
        if self.model.data is None:
            return False
        mask = self.model.get_segment_mask(percentile, continuous)
        return bool(mask[self.model.index])

    def script_next_slice(self: 'MainController') -> None:
        self._next_slice()

    def script_prev_slice(self: 'MainController') -> None:
        self._prev_slice()

    def script_save(self: 'MainController') -> None:
        self._quick_save_masks()

