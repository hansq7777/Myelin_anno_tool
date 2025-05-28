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


def _second_derivative(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.gradient(np.gradient(arr, axis=axis), axis=axis)


def _cross_derivative(arr: np.ndarray) -> np.ndarray:
    return np.gradient(np.gradient(arr, axis=1), axis=0)


def vesselness2d(slice_: np.ndarray, sigmas: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Compute multi-scale vesselness and orientation maps."""
    vesselness = np.zeros_like(slice_, dtype=float)
    orientation = np.zeros_like(slice_, dtype=float)
    for sigma in sigmas:
        sm = gaussian_blur_slice(slice_.astype(float), sigma)
        dxx = _second_derivative(sm, 1) * sigma**2
        dyy = _second_derivative(sm, 0) * sigma**2
        dxy = _cross_derivative(sm) * sigma**2
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy**2)
        lam1 = 0.5 * (dxx + dyy + tmp)
        lam2 = 0.5 * (dxx + dyy - tmp)
        swap = np.abs(lam1) > np.abs(lam2)
        lam1[swap], lam2[swap] = lam2[swap], lam1[swap]
        beta = 0.5
        c = 15.0
        ra = np.abs(lam1) / (np.abs(lam2) + 1e-12)
        rb = np.sqrt(lam1**2 + lam2**2)
        v = (1 - np.exp(-(ra**2) / (2 * beta**2))) * np.exp(-(rb**2) / (2 * c**2))
        v[lam2 > 0] = 0
        ori = 0.5 * np.arctan2(2 * dxy, dxx - dyy)
        update = v > vesselness
        vesselness[update] = v[update]
        orientation[update] = ori[update]
    return vesselness, orientation


def vesselness_region_grow(
    slice_: np.ndarray,
    seeds: np.ndarray,
    sigmas: list[float] | None = None,
    angle_thresh: float = np.pi / 6,
    vesselness_thresh: float = 0.2,
) -> np.ndarray:
    """Grow ``seeds`` using vesselness and orientation consistency."""
    if sigmas is None:
        sigmas = [1, 2, 3]
    vess, orient = vesselness2d(slice_, sigmas)
    mask = seeds.astype(np.uint8).copy()
    visited = mask.astype(bool)
    h, w = mask.shape
    q = [(y, x) for y, x in zip(*np.nonzero(mask))]
    while q:
        y, x = q.pop(0)
        ref_ori = orient[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if vess[ny, nx] >= vesselness_thresh:
                        diff = abs(ref_ori - orient[ny, nx])
                        diff = min(diff, np.pi - diff)
                        if diff <= angle_thresh:
                            visited[ny, nx] = True
                            mask[ny, nx] = 1
                            q.append((ny, nx))
    return mask

