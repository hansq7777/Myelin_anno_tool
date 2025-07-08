import importlib
import sys
from pathlib import Path
import threading
import pytest
np = importlib.import_module('numpy') if importlib.util.find_spec('numpy') else None
if np is None:
    pytest.skip("numpy not available", allow_module_level=True)
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from zstack_anno.utils.morphology_tools import (
    dilate,
    erode,
    label_components,
    remove_small,
    histogram_stretch_stack,
    remove_mask_background,
    remove_mask_background_stack,
    sample_seeds,
    intensity_region_grow,
    flood_region_grow,
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


def test_remove_mask_background_multiple_components():
    img = np.array(
        [
            [20, 20, 0, 80, 80],
            [20, 20, 0, 80, 80],
        ],
        dtype=np.uint8,
    )
    mask = np.array(
        [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    filtered = remove_mask_background(img.astype(float), mask, 20)
    assert np.array_equal(filtered, mask)


def test_remove_mask_background_global_thresh():
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    mask = np.ones_like(img, dtype=np.uint8)
    filtered = remove_mask_background(img, mask, 0, global_thresh=25)
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


def test_flood_region_grow():
    img = np.array(
        [
            [10, 10, 10, 10, 10],
            [10, 50, 50, 50, 10],
            [10, 50, 80, 50, 10],
            [10, 50, 50, 50, 10],
            [10, 10, 10, 10, 10],
        ],
        dtype=float,
    )
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[2, 2] = 1
    grown = flood_region_grow(img, mask, connectivity=1, tolerance=30)
    assert grown.sum() > 1
    assert grown[0, 0] == 0


def test_intensity_region_grow_cancel():
    img = np.array(
        [
            [10, 10, 10, 10, 10],
            [10, 50, 50, 50, 10],
            [10, 50, 80, 50, 10],
            [10, 50, 50, 50, 10],
            [10, 10, 10, 10, 10],
        ],
        dtype=float,
    )
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[2, 2] = 1
    event = threading.Event()
    event.set()
    grown = intensity_region_grow(img, mask, cancel_event=event)
    assert np.array_equal(grown, mask)


def test_flood_region_grow_cancel():
    img = np.array(
        [
            [10, 10, 10, 10, 10],
            [10, 50, 50, 50, 10],
            [10, 50, 80, 50, 10],
            [10, 50, 50, 50, 10],
            [10, 10, 10, 10, 10],
        ],
        dtype=float,
    )
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[2, 2] = 1
    event = threading.Event()
    event.set()
    grown = flood_region_grow(img, mask, cancel_event=event)
    assert np.array_equal(grown, mask)


def test_remove_mask_background_stack_progress():
    imgs = np.array([
        [[10, 20], [30, 40]],
        [[50, 60], [70, 80]],
    ], dtype=np.uint8)
    masks = np.ones_like(imgs, dtype=np.uint8)
    calls: list[tuple[int, int]] = []

    def cb(cur: int, total: int) -> None:
        calls.append((cur, total))

    result = remove_mask_background_stack(
        imgs,
        masks,
        50,
        progress=True,
        progress_fn=cb,
    )
    assert result.shape == imgs.shape
    # first call should be (0,total) and last call (total,total)
    assert calls[0] == (0, len(imgs))
    assert calls[-1] == (len(imgs), len(imgs))


def test_remove_mask_background_stack_parallel():
    imgs = np.array(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
        ],
        dtype=np.uint8,
    )
    masks = np.ones_like(imgs, dtype=np.uint8)

    single = remove_mask_background_stack(imgs, masks, 50)
    parallel = remove_mask_background_stack(imgs, masks, 50, workers=2)

    assert np.array_equal(single, parallel)


def test_remove_mask_background_progress():
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    mask = np.ones_like(img, dtype=np.uint8)
    calls: list[tuple[int, int]] = []

    def cb(cur: int, total: int) -> None:
        calls.append((cur, total))

    result = remove_mask_background(
        img,
        mask,
        50,
        progress=True,
        progress_fn=cb,
    )
    assert result.shape == img.shape
    assert calls[0] == (0, 1)
    assert calls[-1] == (1, 1)



