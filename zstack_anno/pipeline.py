import argparse
import json
import os
from typing import Iterable

import numpy as np
import tifffile

from .models.zstack_model import ZStackModel
from .utils import morphology_tools


class StrategyRunner:
    """Apply script actions to a :class:`ZStackModel` without a GUI."""

    ACTION_MAP = {
        "Dilate": "dilate",
        "Erode": "erode",
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
        "Next Slice": "next_slice",
        "Previous Slice": "prev_slice",
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

    def filter_small(self, threshold: int = 100) -> None:
        self.model.ensure_masks()
        cur = self.model.get_mask()
        new = morphology_tools.remove_small(cur, threshold)
        self.model.set_mask(new)

    def threshold_abs(self, value: float = 0.0) -> None:
        self.model.ensure_masks()
        self.model.threshold_absolute(value)

    def threshold_norm(self, percent: float = 50.0) -> None:
        self.model.ensure_masks()
        self.model.threshold_normalized(percent)

    def seed(self, percentile: float = 90.0) -> None:
        self.model.ensure_masks()
        img = self.model._extract_slice(self.model.index)
        cur = self.model.get_mask()
        seeds = morphology_tools.sample_seeds(img, percentile, num_seeds=20000)
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

    def run_steps(self, steps: Iterable[dict]) -> bool:
        """Run ``steps`` once. Return ``False`` if aborted early."""
        for step in steps:
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


def overlay_image(img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = ZStackModel._normalize_to_8bit(img)
    out = np.stack([base] * 3, axis=-1)
    tp = np.logical_and(gt > 0, pred > 0)
    fn = np.logical_and(gt > 0, pred == 0)
    fp = np.logical_and(gt == 0, pred > 0)
    out[tp] = (0, 255, 0)
    out[fn] = (0, 0, 255)
    out[fp] = (255, 0, 0)
    return out


def run_strategy(path: str, gt_path: str, steps: Iterable[dict]) -> tuple[np.ndarray, float, float]:
    model = ZStackModel()
    model.load(path)
    model.ensure_masks()
    runner = StrategyRunner(model)

    for idx in range(model.n_slices):
        model.index = idx
        runner.run_steps(steps)
        # ensure index stays within bounds
        model.index = min(model.index, model.n_slices - 1)

    pred = model.masks.astype(np.uint8)
    gt = tifffile.imread(gt_path).astype(np.uint8)
    if gt.shape != pred.shape:
        raise ValueError("Ground truth and prediction shape mismatch")
    precision, recall = compute_metrics(gt, pred)
    return pred, precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation strategies")
    parser.add_argument("--stack", required=True, help="Path to image stack")
    parser.add_argument("--groundtruth", required=True, help="Path to ground truth mask stack")
    parser.add_argument("--strategies", nargs="+", required=True, help="JSON files defining strategies")
    parser.add_argument("--output", default="pipeline_results", help="Folder for output overlays")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for strat in args.strategies:
        with open(strat, "r", encoding="utf-8") as f:
            steps = json.load(f)
        pred, precision, recall = run_strategy(args.stack, args.groundtruth, steps)
        name = os.path.splitext(os.path.basename(strat))[0]
        for i in range(pred.shape[0]):
            img = tifffile.imread(args.stack)[i]
            gt = tifffile.imread(args.groundtruth)[i]
            overlay = overlay_image(img, gt, pred[i])
            out_path = os.path.join(args.output, f"{name}_slice{i+1}.png")
            tifffile.imwrite(out_path, overlay)
        print(f"{name}: precision={precision:.3f} recall={recall:.3f}")


if __name__ == "__main__":
    main()
