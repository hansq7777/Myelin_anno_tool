import tifffile
import numpy as np

class ZStackModel:
    def __init__(self):
        self.data: np.ndarray | None = None
        self.index: int = 0

	def load(self, path: str) -> None:
    	self.data = tifffile.imread(path)
    	print("Loaded shape:", self.data.shape, "dtype:", self.data.dtype)  # Log loaded array shape and dtype
    	self.index = 0



    # 便利属性
    @property
    def n_slices(self) -> int:
        return 0 if self.data is None else self.data.shape[0]

    def get_current(self) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("No image loaded")
        return self.data[self.index]
        
    