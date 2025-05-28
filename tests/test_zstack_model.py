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

    # histogram stretch should also work from the original data
    model.histogram_stretch(10)
    expected_stretch = histogram_stretch_stack(arr, 10)
    assert np.array_equal(model.data, expected_stretch)
