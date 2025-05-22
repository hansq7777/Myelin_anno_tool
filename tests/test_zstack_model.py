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
