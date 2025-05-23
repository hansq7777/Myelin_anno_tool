import importlib
import pytest
np = importlib.import_module('numpy') if importlib.util.find_spec('numpy') else None
if np is None:
    pytest.skip("numpy not available", allow_module_level=True)
from zstack_anno.utils.morphology_tools import (
    dilate,
    erode,
    label_components,
    remove_small,
    histogram_stretch_stack,
    remove_mask_background,
)


def test_dilate():
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1
    expected = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    assert np.array_equal(dilate(mask), expected)


def test_erode():
    mask = np.ones((3, 3), dtype=np.uint8)
    mask[0, 0] = 0
    expected = np.zeros((3, 3), dtype=np.uint8)
    assert np.array_equal(erode(mask), expected)


def test_label_components():
    mask = np.array(
        [[1, 0, 0],
         [1, 1, 0],
         [0, 0, 1]],
        dtype=np.uint8,
    )
    labels = label_components(mask)
    assert labels.max() == 2
    assert labels[0, 0] == labels[1, 1]


def test_remove_small():
    mask = np.array(
        [[1, 0, 0],
         [1, 1, 0],
         [0, 0, 1]],
        dtype=np.uint8,
    )
    filtered = remove_small(mask, 3)
    expected = np.array(
        [[1, 0, 0],
         [1, 1, 0],
         [0, 0, 0]],
        dtype=np.uint8,
    )
    assert np.array_equal(filtered, expected)


def test_histogram_stretch_stack():
    stack = np.array([[[0, 10], [20, 30]]], dtype=np.uint8)
    stretched = histogram_stretch_stack(stack, 25)
    assert stretched.shape == stack.shape
    assert stretched.min() == 0
    assert stretched.max() == 255


def test_remove_mask_background():
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    mask = np.ones_like(img, dtype=np.uint8)
    filtered = remove_mask_background(img, mask, 50)
    expected = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    assert np.array_equal(filtered, expected)

