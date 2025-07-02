import numpy as np

try:
    from skimage.morphology import binary_dilation, binary_erosion, remove_small_objects
    from skimage.measure import label
except Exception:  # pragma: no cover - scikit-image may be unavailable
    binary_dilation = None  # type: ignore
    binary_erosion = None  # type: ignore
    remove_small_objects = None  # type: ignore
    label = None  # type: ignore
    gaussian = None  # type: ignore
else:
    try:
        from skimage.filters import gaussian
    except Exception:  # pragma: no cover - scikit-image may be unavailable
        gaussian = None  # type: ignore

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:  # pragma: no cover - scipy may be unavailable
    gaussian_filter = None


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
    if binary_dilation is not None:
        result = binary_dilation(mask > 0, footprint=np.ones((3, 3)), iterations=iterations)
        return result.astype(mask.dtype)
    result = mask.copy()
    for _ in range(iterations):
        result = _dilate_once(result)
    return result


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if binary_erosion is not None:
        result = binary_erosion(mask > 0, footprint=np.ones((3, 3)), iterations=iterations)
        return result.astype(mask.dtype)
    result = mask.copy()
    for _ in range(iterations):
        result = _erode_once(result)
    return result


def label_components(mask: np.ndarray) -> np.ndarray:
    """Label connected components in a binary mask."""
    if label is not None:
        return label(mask > 0, connectivity=1)
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
    if remove_small_objects is not None:
        result = remove_small_objects(mask > 0, min_size=min_size)
        return result.astype(mask.dtype)
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


def _gaussian_blur_slice_numpy(slice_: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur implementation using only NumPy."""
    radius = max(1, int(3 * sigma))
    ax = np.arange(-radius, radius + 1)
    kernel1d = np.exp(-(ax**2) / (2 * sigma**2))
    kernel1d /= kernel1d.sum()
    kernel2d = np.outer(kernel1d, kernel1d)
    pad = np.pad(slice_, radius, mode="edge").astype(float)
    h, w = slice_.shape
    out = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            region = pad[y : y + 2 * radius + 1, x : x + 2 * radius + 1]
            out[y, x] = np.sum(region * kernel2d)
    if np.issubdtype(slice_.dtype, np.integer):
        info = np.iinfo(slice_.dtype)
        out = np.clip(out, 0, info.max)
    return out.astype(slice_.dtype)


def gaussian_blur_slice(slice_: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a single slice with Gaussian kernel."""
    if gaussian_filter is not None:  # pragma: no cover - optional dependency
        return gaussian_filter(slice_, sigma=sigma)
    if gaussian is not None:
        return gaussian(slice_, sigma=sigma, preserve_range=True)
    return _gaussian_blur_slice_numpy(slice_, sigma)


def gaussian_blur_stack(stack: np.ndarray, sigma: float) -> np.ndarray:
    """Apply ``gaussian_blur_slice`` to each slice of a stack."""
    return np.stack([gaussian_blur_slice(s, sigma) for s in stack])


def sample_seeds(
    slice_: np.ndarray, percentile: float, num_seeds: int = 20000
) -> np.ndarray:
    """Randomly sample ``num_seeds`` pixels above a percentile threshold."""
    thresh = np.percentile(slice_, percentile)
    coords = np.argwhere(slice_ > thresh)
    if coords.size == 0:
        return np.zeros_like(slice_, dtype=np.uint8)
    n = min(num_seeds, coords.shape[0])
    idx = np.random.default_rng().choice(coords.shape[0], n, replace=False)
    selected = coords[idx]
    mask = np.zeros_like(slice_, dtype=np.uint8)
    mask[selected[:, 0], selected[:, 1]] = 1
    return mask



def intensity_region_grow(
    slice_: np.ndarray,
    mask: np.ndarray,
    diff_percent: float = 20.0,
    hist_percent: float | None = None,
) -> np.ndarray:
    """Grow ``mask`` based on intensity similarity.

    Pixels are added if their intensity is within ``diff_percent`` of the
    seed region mean. If ``hist_percent`` is provided, pixels below this
    percentile of the slice histogram are ignored.
    """

    labels = label_components(mask)
    unique = np.unique(labels)
    unique = unique[unique != 0]
    h, w = mask.shape
    thresh = None
    if hist_percent is not None:
        thresh = float(np.percentile(slice_, hist_percent))

    for lv in unique:
        region = labels == lv
        if not np.any(region):
            continue
        mean_intensity = float(slice_[region].astype(float).mean())
        diff_thresh = mean_intensity * (diff_percent / 100.0)
        q = [tuple(pt) for pt in zip(*np.nonzero(region))]
        visited = set(q)
        while q:
            y, x = q.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if labels[ny, nx] != 0 or (ny, nx) in visited:
                        continue
                    val = float(slice_[ny, nx])
                    if thresh is not None and val < thresh:
                        continue
                    if abs(val - mean_intensity) <= diff_thresh:
                        labels[ny, nx] = lv
                        visited.add((ny, nx))
                        q.append((ny, nx))

    final = label_components(labels > 0)
    return (final > 0).astype(np.uint8)

