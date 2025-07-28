import sys
import types
from pathlib import Path
import importlib
import tempfile
import numpy as np
import tifffile
import pytest

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

np_spec = importlib.util.find_spec("numpy")
if np_spec is None:
    pytest.skip("numpy not available", allow_module_level=True)
if "tifffile" not in sys.modules:
    sys.modules["tifffile"] = types.ModuleType("tifffile")

from zstack_anno.pipeline import run_strategy, read_stack, grid_search_strategy


def test_run_strategy_single_slice(tmp_path):
    stack = np.zeros((2, 2, 2), dtype=np.uint8)
    gt = np.zeros_like(stack)
    stack[0, 0, 0] = 100
    gt[0, 0, 0] = 1
    stack[1, 0, 1] = 100
    gt[1, 0, 1] = 1

    stack_path = tmp_path / "stack.tif"
    gt_path = tmp_path / "gt.tif"
    tifffile.imwrite(stack_path, stack)
    tifffile.imwrite(gt_path, gt)

    steps = [{"action": "Threshold Abs", "params": {"value": 50}}]
    pred, precision, recall = run_strategy(
        str(stack_path), str(gt_path), steps, slice_idx=1
    )

    assert pred.shape == stack.shape
    assert np.array_equal(pred[1], gt[1])
    assert np.array_equal(pred[0], np.zeros((2, 2), dtype=np.uint8))
    assert precision == 1.0
    assert recall == 1.0


def test_run_strategy_extra_dims(tmp_path):
    stack = np.zeros((1, 1, 2, 2, 2), dtype=np.uint8)
    gt = np.zeros_like(stack)
    stack[0, 0, 1, 0, 1] = 100
    gt[0, 0, 1, 0, 1] = 1

    stack_path = tmp_path / "stack.tif"
    gt_path = tmp_path / "gt.tif"
    tifffile.imwrite(stack_path, stack)
    tifffile.imwrite(gt_path, gt)

    steps = [{"action": "Threshold Abs", "params": {"value": 50}}]
    pred, precision, recall = run_strategy(str(stack_path), str(gt_path), steps)

    assert pred.shape == (2, 2, 2)
    assert np.array_equal(pred[1], gt.squeeze()[1])
    assert precision == 1.0
    assert recall == 1.0


def test_read_stack_squeezes(tmp_path):
    arr = np.zeros((1, 1, 3, 2, 2), dtype=np.uint8)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, arr)

    result = read_stack(str(path))

    assert result.shape == (3, 2, 2)


def test_read_stack_czi(monkeypatch):
    import numpy as np

    def dummy(path):
        return np.zeros((1, 1, 3, 2, 2), dtype=np.uint8), '<xml/>'

    monkeypatch.setattr('zstack_anno.utils.czi_utils.read_czi_stack', dummy)

    result = read_stack('image.czi')
    assert result.shape == (3, 2, 2)


def test_grid_search_strategy(tmp_path):
    stack = np.zeros((2, 2, 2), dtype=np.uint8)
    gt = np.zeros_like(stack)
    stack[0, 0, 0] = 100
    gt[0, 0, 0] = 1

    stack_path = tmp_path / "stack.tif"
    gt_path = tmp_path / "gt.tif"
    tifffile.imwrite(stack_path, stack)
    tifffile.imwrite(gt_path, gt)

    base_steps = [{"action": "Threshold Abs", "params": {"value": 50}}]
    grid = {"1.value": [30, 50]}

    results = grid_search_strategy(
        str(stack_path),
        str(gt_path),
        base_steps,
        grid,
        base_name="thr",
        save_dir=str(tmp_path),
    )

    assert len(results) == 2
    names = [r[0] for r in results]
    assert (tmp_path / "thr_value30.json").exists()
    assert (tmp_path / "thr_value50.json").exists()
    idx = names.index("thr_value50")
    assert np.array_equal(results[idx][1], gt)
