import argparse
import json
import os
import itertools
from typing import Iterable, Callable

import numpy as np
import tifffile

from .models.zstack_model import ZStackModel
from .utils import morphology_tools


class StrategyRunner:
    """Apply script actions to a :class:`ZStackModel` without a GUI."""

    ACTION_MAP = {
        "Dilate": "dilate",
        "Erode": "erode",
        "Close": "close",
        "Filter Small": "filter_small",
        "Threshold Abs": "threshold_abs",
        "Threshold Norm": "threshold_norm",
        "Seed": "seed",
        "Intensity Grow": "int_grow",
        "Flood Grow": "flood_grow",
        "Background Filter": "bg_filter",
        "Histogram Stretch": "stretch",
        "Gaussian Blur": "blur",
        "Clear Blur": "clear_blur",
        "Reverse Intensities": "reverse",
        "Check Segment": "check_segment",
        "Previous Slice": "prev_slice",
        "Felzenszwalb": "felzenszwalb",
        "Watershed IFT": "watershed_ift",
        "scikit-fmm": "scikit_fmm",
        "Fast Marching": "fast_marching",
        "OpenCV Ridge": "opencv_ridge",
        "Steger Ridge": "steger_ridge",
        "Chan-Vese": "chan_vese",
        "CED Filter": "ced_filter",
        "TubeTK Tubes": "tubetk_segment",
        "TubeTK Seed Path": "tubetk_seed_path",
        "Hessian Filter": "hessian_filter",
        "Gabor Filter": "gabor_filter",
        "OpenCV Gabor": "cv_gabor_filter",
        "Structure Tensor": "structure_tensor",
    }

    def __init__(self, model: ZStackModel) -> None:
        self.model = model

    def dilate(self, iterations: int = 1) -> None:
        self.model.ensure_masks()
        cur = self.model.get_mask()
        new = morphology_tools.dilate(cur, iterations=iterations)
        self.model.set_mask(new)

    def erode(self, iterations: int = 1) -> None:
        self.model.ensure_masks()
        cur = self.model.get_mask()
        new = morphology_tools.erode(cur, iterations=iterations)
        self.model.set_mask(new)

    def close(self, strength: int = 1) -> None:
        self.model.ensure_masks()
        cur = self.model.get_mask()
        new = morphology_tools.close(cur, strength)
        self.model.set_mask(new)

    def filter_small(self, threshold: int = 100) -> None:
        self.model.ensure_masks()
        if self.model.masks is None:
            return
        if threshold <= 1:
            return
        new, labels = morphology_tools.remove_small_components(self.model.masks, threshold)
        self.model.replace_masks(new, components=labels, dirty=True)

    def threshold_abs(self, value: float = 0.0) -> None:
        self.model.ensure_masks()
        self.model.threshold_absolute(value)

    def threshold_norm(self, percent: float = 50.0) -> None:
        self.model.ensure_masks()
        self.model.threshold_normalized(percent)

    def seed(
        self, percentile: float = 90.0, pixel_percent: float = 1.0
    ) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        cur = self.model.get_mask()
        seeds = morphology_tools.sample_seeds(
            img, percentile, pixel_percent=pixel_percent
        )
        cur = cur.copy()
        cur[seeds > 0] = 1
        self.model.set_mask(cur)

    def int_grow(
        self,
        diff_pct: float = 20.0,
        hist_pct: float | None = None,
        force_pct: float | None = None,
        limit: int | None = 30000,
    ) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        cur = self.model.get_mask()
        grown = morphology_tools.intensity_region_grow(
            img.astype(float),
            cur,
            diff_pct,
            hist_pct,
            force_pct,
            limit,
        )
        self.model.set_mask(grown)

    def flood_grow(self, connectivity: int = 1, tolerance: float = 5.0) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        cur = self.model.get_mask()
        grown = morphology_tools.flood_region_grow(
            img.astype(float),
            cur,
            connectivity=connectivity,
            tolerance=tolerance,
        )
        self.model.set_mask(grown)

    def blur(self, sigma: float = 1.0) -> None:
        self.model.apply_gaussian_blur(sigma)

    def clear_blur(self) -> None:
        self.model.remove_gaussian_blur()

    def reverse(self) -> None:
        self.model.toggle_reverse_intensity()

    def stretch(self, percentile: float = 0.0) -> None:
        if percentile <= 0:
            self.model.reset_contrast()
        else:
            self.model.histogram_stretch(percentile)

    def bg_filter(self, percentile: float = 10.0, bins: int = 0) -> None:
        self.model.ensure_masks()
        self.model.remove_background(percentile, bins)

    def check_segment(self, percentile: float = 5.0, continuous: bool = True) -> bool:
        mask = self.model.get_segment_mask(percentile, continuous)
        return bool(mask[self.model.index])

    def next_slice(self) -> None:
        if self.model.index + 1 < self.model.n_slices:
            self.model.index += 1

    def prev_slice(self) -> None:
        if self.model.index > 0:
            self.model.index -= 1

    def felzenszwalb(
        self, scale: float = 100.0, sigma: float = 0.8, min_size: int = 20
    ) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        labels = morphology_tools.felzenszwalb_slice(
            img, scale=scale, sigma=sigma, min_size=min_size
        )
        mask = (labels > labels.min()).astype(np.uint8)
        self.model.set_mask(mask)

    def watershed_ift(self) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        markers = morphology_tools.label_components(self.model.get_mask())
        labels = morphology_tools.watershed_ift_slice(img, markers)
        self.model.set_mask((labels > 0).astype(np.uint8))

    def scikit_fmm(self, max_distance: float = 10.0) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        seeds = self.model.get_mask()
        dist = morphology_tools.fmm_distance_slice(img, seeds)
        self.model.set_mask((dist <= max_distance).astype(np.uint8))

    def fast_marching(self, stopping_value: float = 10.0) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        seeds = self.model.get_mask()
        dist = morphology_tools.sitk_fast_marching_slice(
            img, seeds, stopping_value=stopping_value
        )
        self.model.set_mask((dist <= stopping_value).astype(np.uint8))

    def opencv_ridge(self, threshold: float = 0.5) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        resp = morphology_tools.opencv_ridge_filter_slice(img)
        self.model.set_mask((resp > threshold).astype(np.uint8))

    def steger_ridge(self, sigma: float = 1.0, threshold: float = 0.5) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        resp = morphology_tools.steger_ridge_filter_slice(img, sigma=sigma)
        self.model.set_mask((resp > threshold).astype(np.uint8))

    def chan_vese(
        self,
        iterations: int = 50,
        smoothing: int = 1,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
    ) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        mask = morphology_tools.chan_vese_slice(
            img,
            iterations=iterations,
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        self.model.set_mask(mask)

    def ced_filter(self, iterations: int = 5) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        resp = morphology_tools.ced_filter_slice(img, iterations=iterations)
        self.model.set_mask((resp > resp.mean()).astype(np.uint8))

    def tubetk_segment(self) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        mask = morphology_tools.tubetk_segment_tubes_slice(img)
        self.model.set_mask(mask)

    def tubetk_seed_path(self) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        seeds = self.model.get_mask()
        mask = morphology_tools.tubetk_seeded_path_slice(img, seeds)
        self.model.set_mask(mask)

    def hessian_filter(
        self,
        sigma_start: float = 1.0,
        sigma_end: float = 3.0,
        sigma_step: float = 1.0,
        threshold: float = 0.5,
        black_ridges: bool = True,
    ) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        sigmas = np.arange(sigma_start, sigma_end + sigma_step, sigma_step)
        resp = morphology_tools.hessian_filter_slice(
            img, sigmas=sigmas, black_ridges=black_ridges
        )
        self.model.set_mask((resp > threshold).astype(np.uint8))

    def gabor_filter(self, frequency: float = 0.1, theta: float = 0.0) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        resp = morphology_tools.gabor_filter_slice(img, frequency=frequency, theta=theta)
        self.model.set_mask((resp > resp.mean()).astype(np.uint8))

    def cv_gabor_filter(
        self,
        ksize: int = 21,
        sigma: float = 5.0,
        theta: float = 0.0,
        lambd: float = 10.0,
        gamma: float = 0.5,
        psi: float = 0.0,
    ) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        resp = morphology_tools.cv_gabor_filter_slice(
            img, ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi
        )
        self.model.set_mask((resp > resp.mean()).astype(np.uint8))

    def structure_tensor(self, sigma: float = 1.0, threshold: float = 0.5) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        resp = morphology_tools.structure_tensor_eigen_slice(img, sigma=sigma)
        self.model.set_mask((resp > threshold).astype(np.uint8))

    def run_steps(
        self,
        steps: Iterable[dict],
        *,
        callback: Callable[[np.ndarray, int, int], None] | None = None,
    ) -> bool:
        """Run ``steps`` once and optionally invoke ``callback`` after each step.

        Parameters
        ----------
        steps:
            Iterable of step dictionaries as produced by the script editor.
        callback:
            Called with the current mask array, slice index and step index after
            each step if provided.
        """
        for idx, step in enumerate(steps):
            action = step.get("action")
            if not action:
                continue
            method_name = self.ACTION_MAP.get(action)
            if method_name is None:
                continue
            method = getattr(self, method_name, None)
            if method is None:
                continue
            params = step.get("params", {})
            result = method(**params)
            if callback is not None:
                callback(self.model.masks.astype(np.uint8), self.model.index, idx)
            if result is False:
                self.next_slice()
                return False
        return True


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    gt_bin = gt.astype(bool)
    pred_bin = pred.astype(bool)
    tp = np.logical_and(gt_bin, pred_bin).sum()
    fp = np.logical_and(~gt_bin, pred_bin).sum()
    fn = np.logical_and(gt_bin, ~pred_bin).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def read_stack(path: str) -> np.ndarray:
    """Load a stack from ``path`` (TIFF or CZI) and squeeze it to 3-D."""
    if path.lower().endswith('.czi'):
        from .utils.czi_utils import read_czi_stack, CziNotSupportedError

        try:
            arr, _ = read_czi_stack(path)
        except CziNotSupportedError as exc:
            raise RuntimeError(str(exc)) from exc
    else:
        arr = tifffile.imread(path)
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError("Only 3-D stacks are supported")
    return arr


def overlay_image(
    img: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Return RGB overlay image with semi-transparent masks.

    Parameters
    ----------
    img : np.ndarray
        Base grayscale image.
    gt : np.ndarray
        Ground truth mask.
    pred : np.ndarray
        Predicted mask.
    alpha : float, optional
        Opacity of the colored overlay, by default 0.5.
    """

    base = ZStackModel._normalize_to_8bit(img).astype(float)
    out = np.stack([base] * 3, axis=-1)
    overlay = np.zeros_like(out)
    tp = np.logical_and(gt > 0, pred > 0)
    fn = np.logical_and(gt > 0, pred == 0)
    fp = np.logical_and(gt == 0, pred > 0)
    overlay[tp] = (0, 158, 115)  # green
    overlay[fn] = (0, 114, 178)  # blue
    overlay[fp] = (213, 94, 0)   # orange
    mask = tp | fn | fp
    out[mask] = (1 - alpha) * out[mask] + alpha * overlay[mask]
    return out.astype(np.uint8)


def overlay_mask(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Return RGB overlay of ``mask`` on ``img`` without ground truth."""

    base = ZStackModel._normalize_to_8bit(img).astype(float)
    out = np.stack([base] * 3, axis=-1)
    overlay = np.zeros_like(out)
    overlay[mask > 0] = (0, 158, 115)
    out[mask > 0] = (1 - alpha) * out[mask > 0] + alpha * overlay[mask > 0]
    return out.astype(np.uint8)


def run_strategy(
    path: str,
    gt_path: str,
    steps: Iterable[dict],
    *,
    slice_idx: int | None = None,
    step_callback: Callable[[np.ndarray, int, int], None] | None = None,
) -> tuple[np.ndarray, float, float]:
    model = ZStackModel()
    model.load(path)
    model.ensure_masks()
    runner = StrategyRunner(model)

    if slice_idx is None:
        indices = range(model.n_slices)
    else:
        if not (0 <= slice_idx < model.n_slices):
            raise ValueError("slice index out of range")
        indices = [slice_idx]

    for idx in indices:
        model.index = idx

        def cb(mask: np.ndarray, cur_idx: int, step_idx: int) -> None:
            if step_callback is not None:
                step_callback(mask, cur_idx, step_idx)

        runner.run_steps(steps, callback=cb)
        # ensure index stays within bounds
        model.index = min(model.index, model.n_slices - 1)

    pred = model.masks.astype(np.uint8)
    gt = read_stack(gt_path).astype(np.uint8)
    if gt.shape != pred.shape:
        raise ValueError("Ground truth and prediction shape mismatch")
    if slice_idx is not None:
        precision, recall = compute_metrics(gt[slice_idx], pred[slice_idx])
    else:
        precision, recall = compute_metrics(gt, pred)
    return pred, precision, recall


def grid_search_strategy(
    stack_path: str,
    gt_path: str,
    base_steps: Iterable[dict],
    param_grid: dict[str, list],
    *,
    base_name: str = "strategy",
    save_dir: str | None = None,
    slice_idx: int | None = None,
    step_callback: Callable[[np.ndarray, int, int], None] | None = None,
) -> list[tuple[str, np.ndarray, float, float]]:
    """Run ``run_strategy`` for all combinations in ``param_grid``.

    New strategy JSON files are written to ``save_dir`` using ``base_name`` plus
    the parameter values.
    """

    if save_dir is None:
        save_dir = os.getcwd()

    keys = list(param_grid.keys())
    grids = [param_grid[k] for k in keys]
    combos = list(itertools.product(*grids)) if keys else [()]
    results = []
    for values in combos:
        steps = json.loads(json.dumps(list(base_steps)))
        suffix_parts = []
        for key, val in zip(keys, values):
            if "." in key:
                idx_str, param = key.split(".", 1)
                idx = int(idx_str) - 1
            else:
                idx = 0
                param = key
            if idx < len(steps):
                steps[idx].setdefault("params", {})[param] = val
            suffix_parts.append(f"{param}{val}")
        suffix = "_".join(suffix_parts) if suffix_parts else "default"
        name = f"{base_name}_{suffix}"
        script_path = os.path.join(save_dir, name + ".json")
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(steps, f, indent=2)
        pred, prec, rec = run_strategy(
            stack_path,
            gt_path,
            steps,
            slice_idx=slice_idx,
            step_callback=step_callback,
        )
        results.append((name, pred, prec, rec))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation strategies")
    parser.add_argument("--stack", required=True, help="Path to image stack")
    parser.add_argument(
        "--groundtruth", required=True, help="Path to ground truth mask stack"
    )
    parser.add_argument(
        "--strategies", nargs="+", required=True, help="JSON files defining strategies"
    )
    parser.add_argument(
        "--output", default="pipeline_results", help="Folder for output overlays"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    stack = read_stack(args.stack)
    gt_stack = read_stack(args.groundtruth).astype(np.uint8)

    for strat in args.strategies:
        with open(strat, "r", encoding="utf-8") as f:
            steps = json.load(f)
        pred, precision, recall = run_strategy(args.stack, args.groundtruth, steps)
        name = os.path.splitext(os.path.basename(strat))[0]
        for i in range(pred.shape[0]):
            img = stack[i]
            gt = gt_stack[i]
            overlay = overlay_image(img, gt, pred[i])
            out_path = os.path.join(args.output, f"{name}_slice{i+1}.png")
            tifffile.imwrite(out_path, overlay)
        print(f"{name}: precision={precision:.3f} recall={recall:.3f}")


if __name__ == "__main__":
    main()
