import sys
import numpy as np
import warnings
from typing import Callable

try:
    from skimage.morphology import binary_dilation as sk_binary_dilation
    from skimage.morphology import binary_erosion as sk_binary_erosion
    from skimage.morphology import remove_small_objects
    from skimage.morphology import skeletonize
    from skimage.measure import label
except Exception:  # pragma: no cover - scikit-image may be unavailable
    sk_binary_dilation = None  # type: ignore
    sk_binary_erosion = None  # type: ignore
    remove_small_objects = None  # type: ignore
    skeletonize = None  # type: ignore
    label = None  # type: ignore
    gaussian = None  # type: ignore
else:
    try:
        from skimage.filters import gaussian
    except Exception:  # pragma: no cover - scikit-image may be unavailable
        gaussian = None  # type: ignore

try:
    from scipy.ndimage import binary_dilation as nd_binary_dilation
    from scipy.ndimage import binary_erosion as nd_binary_erosion
    from scipy.ndimage import label as nd_label
    from scipy.ndimage import labeled_comprehension
except Exception:  # pragma: no cover - scipy may be unavailable
    nd_binary_dilation = None  # type: ignore
    nd_binary_erosion = None  # type: ignore
    nd_label = None  # type: ignore
    labeled_comprehension = None  # type: ignore

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:  # pragma: no cover - scipy may be unavailable
    gaussian_filter = None


if nd_binary_dilation is None or sk_binary_dilation is None:
    warnings.warn(
        "scipy and/or scikit-image not available; using slower NumPy "
        "fallbacks for morphology operations",
        RuntimeWarning,
    )


def _print_progress(
    prefix: str,
    current: int,
    total: int,
    callback: Callable | None = None,
    *,
    line: int | None = None,
) -> None:
    """Print or callback progress information.

    Parameters
    ----------
    prefix:
        Text shown before the progress bar.
    current, total:
        Current and total iteration counts.
    callback:
        Optional callable used instead of printing.
    line:
        If provided, update the specified console line to allow multiple
        progress bars to be shown in parallel.
    """

    if callback is not None:
        callback(current, total)
        return

    bar_len = 20
    filled = int(bar_len * current / float(total)) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    msg = f"{prefix} [{bar}] {current}/{total}"

    if line is not None:
        sys.stdout.write("\x1b7")  # save cursor
        sys.stdout.write(f"\x1b[{line}F")  # move up
        sys.stdout.write("\r" + msg + "\n")
        sys.stdout.write("\x1b8")  # restore
    else:
        sys.stdout.write("\r" + msg)
        if current == total:
            sys.stdout.write("\n")
    sys.stdout.flush()


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
    """Dilate ``mask`` by a 3x3 structuring element."""
    mask_bool = mask > 0
    if nd_binary_dilation is not None:
        result = nd_binary_dilation(mask_bool, structure=np.ones((3, 3)), iterations=iterations)
        return result.astype(mask.dtype)
    if sk_binary_dilation is not None:
        result = mask_bool
        for _ in range(iterations):
            result = sk_binary_dilation(result, footprint=np.ones((3, 3)))
        return result.astype(mask.dtype)
    result = mask.copy()
    for _ in range(iterations):
        result = _dilate_once(result)
    return result


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode ``mask`` by a 3x3 structuring element."""
    mask_bool = mask > 0
    if nd_binary_erosion is not None:
        result = nd_binary_erosion(mask_bool, structure=np.ones((3, 3)), iterations=iterations)
        return result.astype(mask.dtype)
    if sk_binary_erosion is not None:
        result = mask_bool
        for _ in range(iterations):
            result = sk_binary_erosion(result, footprint=np.ones((3, 3)))
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
    image: np.ndarray,
    mask: np.ndarray,
    percentile: float,
    *,
    progress: bool = False,
    progress_fn: Callable | None = None,
) -> np.ndarray:
    """Remove lowest intensity pixels within ``mask`` based on percentile.

    The threshold is computed independently for each connected component in the
    mask. Pixels are removed from a component if their value is strictly below
    the percentile of that component's intensities. If ``mask`` contains no
    pixels, a copy of ``mask`` is returned unchanged.

    Parameters
    ----------
    image, mask:
        Input slice and corresponding mask.
    percentile:
        Intensity percentile threshold.
    progress:
        If ``True``, report progress for each connected component.
    progress_fn:
        Optional callback for progress updates.
    """

    values = image[mask > 0]
    if values.size == 0:
        return mask.copy()

    if nd_label is None or labeled_comprehension is None:
        labels = label_components(mask)
        result = mask.copy()
        total = labels.max()
        if progress:
            _print_progress("BG filter", 0, total, callback=progress_fn)
        for idx, lbl in enumerate(range(1, total + 1), start=1):
            if progress:
                _print_progress("BG filter", idx, total, callback=progress_fn)
            region = labels == lbl
            if not np.any(region):
                continue
            thresh = np.percentile(image[region], percentile)
            result[region & (image < thresh)] = 0
        return result

    labels, num = nd_label(mask > 0)
    if num == 0:
        return mask.copy()
    if progress:
        _print_progress("BG filter", 0, num, callback=progress_fn)

    indices = np.arange(1, num + 1)
    thresholds = labeled_comprehension(
        image,
        labels,
        indices,
        lambda x: np.percentile(x, percentile),
        float,
        0,
    )
    label_thresholds = np.zeros(num + 1, dtype=float)
    label_thresholds[1:] = thresholds
    threshold_map = label_thresholds[labels]

    result = mask.copy()
    result[(labels > 0) & (image < threshold_map)] = 0

    if progress:
        _print_progress("BG filter", num, num, callback=progress_fn)
    return result


def remove_mask_background_stack(
    images: np.ndarray,
    masks: np.ndarray,
    percentile: float,
    *,
    progress: bool = False,
    progress_fn: Callable | None = None,
    workers: int = 1,
) -> np.ndarray:
    """Apply ``remove_mask_background`` on each slice pair of images and masks.

    Parameters
    ----------
    images, masks:
        Stack of images and corresponding masks.
    percentile:
        Pixels strictly below this percentile within each connected component
        are removed.
    progress:
        If ``True``, display a simple progress bar.
    progress_fn:
        Optional callback invoked with ``(current, total)`` for progress
        reporting.
    workers:
        Number of worker threads to use. Values greater than ``1`` enable
        parallel processing of slices.
    """

    total = len(images)
    if progress:
        _print_progress("BG filter", 0, total, callback=progress_fn)

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        import threading

        results: list[np.ndarray | None] = [None] * total
        count = 0
        lock = threading.Lock()

        def task(args: tuple[int, np.ndarray, np.ndarray]) -> None:
            nonlocal count
            idx, img, msk = args
            res = remove_mask_background(img, msk, percentile)
            results[idx] = res
            if progress:
                with lock:
                    count += 1
                    _print_progress("BG filter", count, total, callback=progress_fn)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            ex.map(task, [(i, im, mk) for i, (im, mk) in enumerate(zip(images, masks))])

        return np.stack([r for r in results if r is not None])

    else:
        result = []
        for idx, (img, msk) in enumerate(zip(images, masks), start=1):
            if progress:
                _print_progress("BG filter", idx, total, callback=progress_fn)
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
    progress: bool = False,
    progress_fn: Callable | None = None,
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

    total = len(unique)
    if progress:
        _print_progress("Int grow", 0, total, callback=progress_fn)
    for idx, lv in enumerate(unique, start=1):
        if progress:
            _print_progress("Int grow", idx, total, callback=progress_fn)
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


def _skeletonize_numpy(slice_: np.ndarray) -> np.ndarray:
    """Fallback skeletonization using erosion/dilation."""
    img = slice_.astype(np.uint8)
    skeleton = np.zeros_like(img)
    working = img.copy()
    while np.any(working):
        eroded = erode(working)
        opened = dilate(eroded)
        temp = working & (~opened)
        skeleton |= temp
        working = eroded
    return skeleton


def _neighbor_count(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    h, w = arr.shape
    count = np.zeros_like(arr, dtype=int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            count += padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return count


