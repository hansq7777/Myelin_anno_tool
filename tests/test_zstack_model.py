import sys
import types
from pathlib import Path
import importlib
import pytest

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Provide minimal stubs for numpy and tifffile if they are missing
np_spec = importlib.util.find_spec("numpy")
if np_spec is None:
    pytest.skip("numpy not available", allow_module_level=True)
if "tifffile" not in sys.modules:
    sys.modules["tifffile"] = types.ModuleType("tifffile")

from zstack_anno.models.zstack_model import ZStackModel
from zstack_anno.pipeline import StrategyRunner
from zstack_anno.utils.morphology_tools import (
    gaussian_blur_stack,
    histogram_stretch_stack,
)
import numpy as np


class DummyArray:
    def __init__(self, data):
        self._data = data
        self.shape = (len(data), len(data[0]), len(data[0][0]))
        self.ndim = 3
        self.dtype = "uint8"

    def __getitem__(self, idx):
        return self._data[idx]


def test_n_slices_and_get_current():
    arr = DummyArray(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )
    model = ZStackModel()
    model.data = arr
    model.index = 0

    assert model.n_slices == 2
    assert model.get_current() == [[1, 2], [3, 4]]


def test_set_and_get_mask():
    model = ZStackModel()
    model.data = np.zeros((2, 1, 1), dtype=np.uint8)
    mask0 = np.array([[1]], dtype=np.uint8)
    mask1 = np.array([[0]], dtype=np.uint8)
    model.set_mask(mask0, slice_idx=0)
    model.set_mask(mask1, slice_idx=1)

    assert np.array_equal(model.get_mask(0), mask0)
    assert np.array_equal(model.get_mask(1), mask1)


def test_update_components():
    model = ZStackModel()
    model.data = np.zeros((1, 3, 3), dtype=np.uint8)
    mask = np.array(
        [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
        dtype=np.uint8,
    )
    model.set_mask(mask, slice_idx=0)
    labels = model.components[0]
    assert labels.max() == 2


def test_counts():
    model = ZStackModel()
    model.data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask0 = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    mask1 = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    model.set_mask(mask0, slice_idx=0)
    model.set_mask(mask1, slice_idx=1)
    assert model.total_pixel_count() == 4
    assert model.component_count() >= 1


def test_blur_and_stretch_are_absolute():
    model = ZStackModel()
    arr = np.arange(16, dtype=np.uint8).reshape(1, 4, 4)
    model.data = arr.copy()
    model.original_data = arr.copy()

    # apply strong blur first
    model.apply_gaussian_blur(2.0)
    first_blur = model.data.copy()

    # weaker blur should be computed from the original image
    model.apply_gaussian_blur(0.5)
    expected = gaussian_blur_stack(arr, 0.5)
    assert np.array_equal(model.data, expected)
    assert not np.array_equal(model.data, first_blur)

    # histogram stretch should work on the blurred data
    model.histogram_stretch(10)
    expected_stretch = histogram_stretch_stack(gaussian_blur_stack(arr, 0.5), 10)
    assert np.array_equal(model.data, expected_stretch)


def test_threshold_absolute_overlay():
    model = ZStackModel()
    model.data = np.array([[[0, 5], [10, 15]]], dtype=np.uint8)
    model.ensure_masks()
    model.masks[0, 0, 0] = 1
    model.threshold_absolute(5, slice_idx=0)
    expected = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    assert np.array_equal(model.get_mask(0), expected)


def test_threshold_normalized_overlay():
    model = ZStackModel()
    model.data = np.array([[[0, 50], [100, 150]]], dtype=np.uint8)
    model.ensure_masks()
    model.masks[0, 0, 0] = 1
    model.threshold_normalized(50.0, slice_idx=0)
    expected = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    assert np.array_equal(model.get_mask(0), expected)


def test_strategy_runner_skip_on_check_segment():
    model = ZStackModel()
    slice0 = np.zeros((10, 10), dtype=np.uint8)
    slice1 = np.zeros((10, 10), dtype=np.uint8)
    slice1.flat[:10] = 100
    model.data = np.stack([slice0, slice1])
    model.original_data = model.data.copy()
    model.ensure_masks()
    runner = StrategyRunner(model)
    steps = [
        {"action": "Check Segment", "params": {"percentile": 50.0, "continuous": True}},
        {"action": "Dilate", "params": {"iterations": 1}},
    ]

    model.index = 0
    result = runner.run_steps(steps)

    assert result is False
    assert model.index == 1
    assert np.array_equal(model.get_mask(0), np.zeros((10, 10), dtype=np.uint8))


def test_toggle_reverse_intensity():
    model = ZStackModel()
    arr = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
    model.data = arr.copy()
    model.original_data = arr.copy()

    model.toggle_reverse_intensity()
    expected = arr.max() - arr
    assert np.array_equal(model.data, expected)

    model.toggle_reverse_intensity()
    assert np.array_equal(model.data, arr)
