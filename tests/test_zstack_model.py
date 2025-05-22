import sys
import types

# Provide minimal stubs for numpy and tifffile if they are missing
if 'numpy' not in sys.modules:
    np_stub = types.ModuleType('numpy')
    np_stub.ndarray = object
    sys.modules['numpy'] = np_stub
if 'tifffile' not in sys.modules:
    sys.modules['tifffile'] = types.ModuleType('tifffile')

from zstack_anno.models.zstack_model import ZStackModel
import numpy as np

class DummyArray:
    def __init__(self, data):
        self._data = data
        self.shape = (len(data), len(data[0]), len(data[0][0]))
        self.ndim = 3
        self.dtype = 'uint8'

    def __getitem__(self, idx):
        return self._data[idx]


def test_n_slices_and_get_current():
    arr = DummyArray([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])
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

