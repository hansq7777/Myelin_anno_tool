import numpy as np


def _apply_single(mask: np.ndarray, func) -> np.ndarray:
    """Apply a morphological function to a single 2-D mask."""
    result = mask
    for _ in range(1):
        result = func(result)
    return result


def _dilate_once(arr: np.ndarray) -> np.ndarray:
    """Fast dilation using vectorised shifts."""
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    h, w = arr.shape
    out = np.zeros_like(arr)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out = np.maximum(out, padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w])
    return out


def _erode_once(arr: np.ndarray) -> np.ndarray:
    """Fast erosion using vectorised shifts."""
    padded = np.pad(arr, 1, mode="constant", constant_values=1)
    h, w = arr.shape
    out = np.ones_like(arr)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out = np.minimum(out, padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w])
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


def label_components(mask: np.ndarray) -> np.ndarray:
    """Label connected components in a binary mask."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                current += 1
                stack = [(y, x)]
                labels[y, x] = current
                while stack:
                    cy, cx = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if (
                                0 <= ny < h
                                and 0 <= nx < w
                                and mask[ny, nx]
                                and labels[ny, nx] == 0
                            ):
                                labels[ny, nx] = current
                                stack.append((ny, nx))
    return labels


def dilate_stack(stack: np.ndarray, iterations: int = 1) -> np.ndarray:
    return np.stack([dilate(slice_, iterations) for slice_ in stack])


def erode_stack(stack: np.ndarray, iterations: int = 1) -> np.ndarray:
    return np.stack([erode(slice_, iterations) for slice_ in stack])

