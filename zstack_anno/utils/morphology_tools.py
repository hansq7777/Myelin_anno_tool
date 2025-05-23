import numpy as np


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


def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than ``min_size``."""
    labels = label_components(mask)
    if labels.max() == 0:
        return mask.copy()
    result = mask.copy()
    for lbl in range(1, labels.max() + 1):
        if np.sum(labels == lbl) < min_size:
            result[labels == lbl] = 0
    return result


def remove_small_stack(stack: np.ndarray, min_size: int) -> np.ndarray:
    """Apply ``remove_small`` to every slice of a stack."""
    return np.stack([remove_small(slice_, min_size) for slice_ in stack])


def histogram_stretch(slice_: np.ndarray, percentile: float) -> np.ndarray:
    """Stretch contrast of a slice using percentile exclusion."""
    low = np.percentile(slice_, percentile)
    high = np.percentile(slice_, 100 - percentile)
    if high <= low:
        return slice_.copy()
    scaled = (slice_ - low) / (high - low)
    scaled = np.clip(scaled, 0, 1)
    if np.issubdtype(slice_.dtype, np.integer):
        info = np.iinfo(slice_.dtype)
        scaled = (scaled * info.max).astype(slice_.dtype)
    else:
        scaled = scaled.astype(slice_.dtype)
    return scaled


def histogram_stretch_stack(stack: np.ndarray, percentile: float) -> np.ndarray:
    """Apply ``histogram_stretch`` to every slice of a stack."""
    return np.stack([histogram_stretch(s, percentile) for s in stack])


def remove_mask_background(
    image: np.ndarray, mask: np.ndarray, percentile: float
) -> np.ndarray:
    """Remove lowest intensity pixels within ``mask`` based on percentile."""
    values = image[mask > 0]
    if values.size == 0:
        return mask.copy()
    thresh = np.percentile(values, percentile)
    result = mask.copy()
    result[(mask > 0) & (image <= thresh)] = 0
    return result


def remove_mask_background_stack(
    images: np.ndarray, masks: np.ndarray, percentile: float
) -> np.ndarray:
    """Apply ``remove_mask_background`` on each slice pair of images and masks."""
    result = []
    for img, msk in zip(images, masks):
        result.append(remove_mask_background(img, msk, percentile))
    return np.stack(result)

