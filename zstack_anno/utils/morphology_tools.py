import numpy as np


def _apply_single(mask: np.ndarray, func) -> np.ndarray:
    """Apply a morphological function to a single 2-D mask."""
    result = mask
    for _ in range(1):
        result = func(result)
    return result


def _dilate_once(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            window = padded[i:i+3, j:j+3]
            out[i, j] = 1 if np.any(window) else 0
    return out


def _erode_once(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, 1, mode="constant", constant_values=1)
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            window = padded[i:i+3, j:j+3]
            out[i, j] = 1 if np.all(window) else 0
    return out


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        result = _dilate_once(result)
    return result


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        result = _erode_once(result)
    return result


def dilate_stack(stack: np.ndarray, iterations: int = 1) -> np.ndarray:
    return np.stack([dilate(slice_, iterations) for slice_ in stack])


def erode_stack(stack: np.ndarray, iterations: int = 1) -> np.ndarray:
    return np.stack([erode(slice_, iterations) for slice_ in stack])

