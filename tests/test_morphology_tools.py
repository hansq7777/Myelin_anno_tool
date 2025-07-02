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
    sample_seeds,
    intensity_region_grow,
    filter_linear_components,
    filter_linear_components_stack,
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


def test_sample_seeds():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    seeds = sample_seeds(img, 90, num_seeds=5)
    assert seeds.sum() == 5
    assert np.all(img[seeds > 0] > np.percentile(img, 90))


def test_intensity_region_grow():
    img = np.array(
        [[10, 10, 10, 10, 10],
         [10, 50, 50, 50, 10],
         [10, 50, 80, 50, 10],
         [10, 50, 50, 50, 10],
         [10, 10, 10, 10, 10]],
        dtype=float,
    )
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[2, 2] = 1
    grown = intensity_region_grow(img, mask, diff_percent=50, hist_percent=20)
    assert grown.sum() > 1
    assert grown[0, 0] == 0


def test_filter_linear_components():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:9, 5] = 1
    mask[2:5, 2:5] = 1
    filtered = filter_linear_components(mask, linearity_thresh=2.0)
    assert filtered[1:9, 5].sum() == 8
    assert filtered[2:5, 2:5].sum() == 0


def test_filter_linear_components_stack():
    stack = np.zeros((3, 6, 6), dtype=np.uint8)
    stack[0, 2, 3] = 1
    stack[1, 2, 3] = 1
    stack[2, 2, 3] = 1
    stack[0:2, 0:2, 0:2] = 1
    result = filter_linear_components_stack(stack, linearity_thresh=2.0)
    assert np.all(result[:, 2, 3] == 1)
    assert result[0:2, 0:2, 0:2].sum() == 0


def test_filter_linear_components_stack_keep_isotropic():
    stack = np.zeros((2, 3, 3), dtype=np.uint8)
    stack[:, 1, 1] = 1
    stack[:, 0, 0] = 1
    result = filter_linear_components_stack(
        stack, linearity_thresh=2.0, require_3d_linearity=False
    )
    assert np.array_equal(result, stack)

