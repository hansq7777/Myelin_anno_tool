import sys
import time
import numpy as np
import warnings
import logging
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from skimage.morphology import (
        binary_dilation as sk_binary_dilation,
        binary_erosion as sk_binary_erosion,
        remove_small_objects,
        skeletonize,
        medial_axis,
    )
    from skimage.measure import label
    from skimage.segmentation import flood, morphological_chan_vese
except ImportError as exc:  # pragma: no cover - scikit-image may be unavailable
    logger.warning("scikit-image import failed: %s", exc)
    sk_binary_dilation = None  # type: ignore
    sk_binary_erosion = None  # type: ignore
    remove_small_objects = None  # type: ignore
    skeletonize = None  # type: ignore
    medial_axis = None  # type: ignore
    label = None  # type: ignore
    gaussian = None  # type: ignore
    flood = None  # type: ignore
    morphological_chan_vese = None  # type: ignore
else:
    try:
        from skimage.filters import gaussian, frangi, sato, meijering, hessian, gabor
        from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    except ImportError as exc:  # pragma: no cover - scikit-image may be unavailable
        logger.warning("scikit-image.filters import failed: %s", exc)
        gaussian = None  # type: ignore
        frangi = None  # type: ignore
        sato = None  # type: ignore
        meijering = None  # type: ignore
        hessian = None  # type: ignore
        gabor = None  # type: ignore
        structure_tensor = None  # type: ignore
        structure_tensor_eigenvalues = None  # type: ignore

skeletonize_3d = None


def _load_skeletonize_3d() -> Callable | None:
    """Attempt to import ``skeletonize_3d`` only when needed."""
    global skeletonize_3d
    if skeletonize_3d is None:
        try:  # pragma: no cover - optional dependency
            from skimage.morphology import skeletonize_3d as _sk_skeletonize_3d
        except Exception as exc:
            logger.warning("skeletonize_3d import failed: %s", exc)
            skeletonize_3d = None
        else:
            skeletonize_3d = _sk_skeletonize_3d
    return skeletonize_3d


try:
    from scipy.ndimage import binary_dilation as nd_binary_dilation
    from scipy.ndimage import binary_erosion as nd_binary_erosion
    from scipy.ndimage import binary_fill_holes as nd_binary_fill_holes
    from scipy.ndimage import label as nd_label
    from scipy.ndimage import labeled_comprehension
except ImportError as exc:  # pragma: no cover - scipy may be unavailable
    logger.warning("scipy.ndimage import failed: %s", exc)
    nd_binary_dilation = None  # type: ignore
    nd_binary_erosion = None  # type: ignore
    nd_binary_fill_holes = None  # type: ignore
    nd_label = None  # type: ignore
    labeled_comprehension = None  # type: ignore

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except ImportError as exc:  # pragma: no cover - scipy may be unavailable
    logger.warning("scipy.ndimage gaussian_filter import failed: %s", exc)
    gaussian_filter = None


if nd_binary_dilation is None or sk_binary_dilation is None:
    warnings.warn(
        "scipy and/or scikit-image not available; using slower NumPy "
        "fallbacks for morphology operations",
        RuntimeWarning,
    )

_START_TIMES: dict[tuple[str, int | None], float] = {}


def _print_progress(
    prefix: str,
    current: int,
    total: int,
    callback: Callable | None = None,
    *,
    line: int | None = None,
    mask: np.ndarray | None = None,
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

    Displays an estimated remaining time when printed to the console.
    """

    key = (prefix, line)
    if callback is not None:
        try:
            callback(current, total, mask)
        except TypeError:
            callback(current, total)  # type: ignore[arg-type]
        if current == 0:
            _START_TIMES.pop(key, None)
        return
    if current == 0:
        _START_TIMES[key] = time.monotonic()

    bar_len = 20
    filled = int(bar_len * current / float(total)) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)

    start = _START_TIMES.get(key)
    eta_msg = ""
    if start is not None:
        elapsed = time.monotonic() - start
        if current >= total:
            eta_msg = f" time {elapsed:.1f}s"
            _START_TIMES.pop(key, None)
        elif current > 0:
            rate = elapsed / current
            eta = rate * (total - current)
            eta_msg = f" ETA {eta:.1f}s"

    msg = f"{prefix} [{bar}] {current}/{total}{eta_msg}"

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
        result = nd_binary_dilation(
            mask_bool, structure=np.ones((3, 3)), iterations=iterations
        )
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
        result = nd_binary_erosion(
            mask_bool, structure=np.ones((3, 3)), iterations=iterations
        )
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
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
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


def close(mask: np.ndarray, strength: int = 1) -> np.ndarray:
    """Fill holes within ``mask`` up to ``strength`` pixels in size.

    ``strength`` controls the maximum area of a hole that will be closed. The
    operation is performed on each connected component individually so that
    neighbouring masks are not merged.
    """

    mask_bool = mask > 0

    # First fill all holes using scipy if available or the fallback below
    if nd_binary_fill_holes is not None:
        filled = nd_binary_fill_holes(mask_bool)
    else:
        inv = ~mask_bool
        labels = label_components(inv.astype(np.uint8))
        if labels.max() == 0:
            return mask.copy()
        border = np.unique(
            np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        )
        fill = ~np.isin(labels, border)
        inv[fill] = False
        filled = ~inv

    # Identify newly filled holes
    holes = filled & (~mask_bool)
    labels = label_components(holes.astype(np.uint8))
    if labels.max() == 0:
        return filled.astype(mask.dtype)

    max_hole_size = max(1, int(strength))
    result = mask_bool.copy()
    for lbl in range(1, labels.max() + 1):
        if np.sum(labels == lbl) <= max_hole_size:
            result[labels == lbl] = True

    return result.astype(mask.dtype)


def close_stack(stack: np.ndarray, strength: int = 1) -> np.ndarray:
    """Apply :func:`close` to every slice of ``stack`` with the given strength."""
    return np.stack([close(s, strength=strength) for s in stack])


def threshold_absolute(slice_: np.ndarray, value: float) -> np.ndarray:
    """Return binary mask of pixels greater than or equal to ``value``."""
    return (slice_.astype(float) >= value).astype(np.uint8)


def threshold_normalized(slice_: np.ndarray, percent: float) -> np.ndarray:
    """Threshold slice after normalizing to 0-1 range by ``percent``."""
    arr = slice_.astype(float)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros_like(slice_, dtype=np.uint8)
    norm = (arr - mn) / (mx - mn)
    return (norm >= (percent / 100.0)).astype(np.uint8)


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
    global_thresh: float | None = None,
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
        Pixels strictly below this percentile within each connected component are
        removed.
    global_thresh:
        Optional intensity threshold computed from the stack histogram. Pixels
        below this value are removed before applying the percentile filter.
    progress:
        If ``True``, report progress for each connected component.
    progress_fn:
        Optional callback for progress updates.
    """

    values = image[mask > 0]
    if values.size == 0:
        return mask.copy()

    work_mask = mask.copy()
    if global_thresh is not None:
        work_mask[(work_mask > 0) & (image < global_thresh)] = 0
        if not np.any(work_mask):
            return work_mask

    if nd_label is None or labeled_comprehension is None:
        labels = label_components(work_mask)
        result = work_mask.copy()
        total = labels.max()
        if progress:
            _print_progress(
                "BG filter",
                0,
                total,
                callback=progress_fn,
                mask=result.copy(),
            )
        for idx, lbl in enumerate(range(1, total + 1), start=1):
            if progress:
                _print_progress(
                    "BG filter",
                    idx,
                    total,
                    callback=progress_fn,
                    mask=result.copy(),
                )
            region = labels == lbl
            if not np.any(region):
                continue
            thresh = np.percentile(image[region], percentile)
            result[region & (image < thresh)] = 0
        return result

    labels, num = nd_label(work_mask > 0)
    if num == 0:
        return work_mask.copy()
    if progress:
        _print_progress(
            "BG filter",
            0,
            num,
            callback=progress_fn,
            mask=work_mask.copy(),
        )

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

    result = work_mask.copy()
    result[(labels > 0) & (image < threshold_map)] = 0

    if progress:
        _print_progress(
            "BG filter",
            num,
            num,
            callback=progress_fn,
            mask=result.copy(),
        )
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
        _print_progress(
            "BG filter",
            0,
            total,
            callback=progress_fn,
            mask=masks[0].copy() if len(masks) else None,
        )

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
                    _print_progress(
                        "BG filter",
                        count,
                        total,
                        callback=progress_fn,
                        mask=res if res is not None else None,
                    )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            ex.map(task, [(i, im, mk) for i, (im, mk) in enumerate(zip(images, masks))])

        return np.stack([r for r in results if r is not None])

    else:
        result = []
        for idx, (img, msk) in enumerate(zip(images, masks), start=1):
            if progress:
                _print_progress(
                    "BG filter",
                    idx,
                    total,
                    callback=progress_fn,
                    mask=result[-1].copy() if result else None,
                )
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
    force_percent: float | None = None,
    max_growth: int | None = None,
    progress: bool = False,
    progress_fn: Callable | None = None,
    cancel_event: "threading.Event | None" = None,
) -> np.ndarray:
    """Grow ``mask`` based on intensity similarity.

    Pixels are added to the region when their intensity is within
    ``diff_percent`` of the current seed pixel.  When ``hist_percent`` is
    provided, a global threshold is computed from the slice histogram and
    pixels below this value are ignored completely.  If ``force_percent`` is
    given, pixels brighter than the corresponding percentile are always added
    regardless of ``diff_percent``.

    If ``cancel_event`` is provided and set during execution, the original
    ``mask`` is returned unchanged. When ``max_growth`` is given, no more than
    this number of pixels will be added to any single component.
    """

    labels = label_components(mask)
    unique = np.unique(labels)
    unique = unique[unique != 0]
    h, w = mask.shape
    thresh = None
    force_thresh = None
    if hist_percent is not None:
        thresh = float(np.percentile(slice_, hist_percent))
    if force_percent is not None:
        force_thresh = float(np.percentile(slice_, force_percent))

    total = len(unique)
    if progress:
        _print_progress(
            "Int grow",
            0,
            total,
            callback=progress_fn,
            mask=(labels > 0).astype(np.uint8),
        )
    for idx, lv in enumerate(unique, start=1):
        if cancel_event is not None and cancel_event.is_set():
            return mask
        if progress:
            _print_progress(
                "Int grow",
                idx,
                total,
                callback=progress_fn,
                mask=(labels > 0).astype(np.uint8),
            )
        region = labels == lv
        if not np.any(region):
            continue
        q = [tuple(pt) for pt in zip(*np.nonzero(region))]
        visited = set(q)
        processed = 0
        region_total = int(region.sum())
        added = 0
        reached = False
        while q:
            if cancel_event is not None and cancel_event.is_set():
                return mask
            y, x = q.pop()
            seed_val = float(slice_[y, x])
            diff_thresh = seed_val * (diff_percent / 100.0)
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
                    if force_thresh is not None and val >= force_thresh:
                        labels[ny, nx] = lv
                        visited.add((ny, nx))
                        q.append((ny, nx))
                        added += 1
                        if max_growth is not None and added >= max_growth:
                            reached = True
                            break
                        continue
                    if abs(val - seed_val) <= diff_thresh:
                        labels[ny, nx] = lv
                        visited.add((ny, nx))
                        q.append((ny, nx))
                        added += 1
                        if max_growth is not None and added >= max_growth:
                            reached = True
                            break
                if reached:
                    break
            if reached:
                break
            processed += 1
            if progress and processed % 5000 == 0 and progress_fn is not None:
                # update the UI without spamming the console
                try:
                    progress_fn(processed, region_total, (labels > 0).astype(np.uint8))
                except TypeError:
                    progress_fn(processed, region_total)  # type: ignore[arg-type]
            if reached:
                break

    final = label_components(labels > 0)
    return (final > 0).astype(np.uint8)


def flood_region_grow(
    slice_: np.ndarray,
    mask: np.ndarray,
    connectivity: int = 1,
    tolerance: float = 5.0,
    *,
    progress: bool = False,
    progress_fn: Callable | None = None,
    workers: int = 1,
    cancel_event: "threading.Event | None" = None,
) -> np.ndarray:
    """Grow ``mask`` using ``skimage.segmentation.flood`` for each component.

    If ``cancel_event`` is provided and set, the partially grown result so far
    is returned.
    """

    def _flood_fallback(seed: tuple[int, int]) -> np.ndarray:
        """Simple flood fill using NumPy when skimage is unavailable."""
        h, w = slice_.shape
        result = np.zeros_like(mask, dtype=bool)
        seed_val = slice_[seed]
        stack = [seed]
        result[seed] = True
        if connectivity > 1:
            offsets = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        else:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while stack:
            cy, cx = stack.pop()
            for dy, dx in offsets:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not result[ny, nx]:
                    if abs(float(slice_[ny, nx]) - float(seed_val)) <= tolerance:
                        result[ny, nx] = True
                        stack.append((ny, nx))
        return result

    labels = label_components(mask)
    unique = np.unique(labels)
    unique = unique[unique != 0]
    result = np.zeros_like(mask, dtype=np.uint8)

    def run_one(lv: int) -> np.ndarray | None:
        region = labels == lv
        if not np.any(region):
            return None
        seed = tuple(np.argwhere(region)[0])
        if flood is not None:
            return flood(slice_, seed, connectivity=connectivity, tolerance=tolerance)
        return _flood_fallback(seed)

    total = len(unique)
    if progress:
        _print_progress(
            "Flood",
            0,
            total,
            callback=progress_fn,
            mask=result.copy(),
        )

    if workers and workers > 1 and total > 1:
        from concurrent.futures import ThreadPoolExecutor
        import threading

        results: list[np.ndarray | None] = [None] * total
        count = 0
        lock = threading.Lock()

        def task(args: tuple[int, int]) -> None:
            nonlocal count
            idx, lv = args
            if cancel_event is not None and cancel_event.is_set():
                return
            res = run_one(lv)
            results[idx] = res
            if progress:
                with lock:
                    count += 1
                    _print_progress(
                        "Flood",
                        count,
                        total,
                        callback=progress_fn,
                        mask=result.copy(),
                    )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            ex.map(task, [(i, lv) for i, lv in enumerate(unique)])

        if cancel_event is not None and cancel_event.is_set():
            return mask
        for res in results:
            if res is not None:
                result[res] = 1
    else:
        for idx, lv in enumerate(unique, start=1):
            if cancel_event is not None and cancel_event.is_set():
                return mask
            if progress:
                _print_progress(
                    "Flood",
                    idx,
                    total,
                    callback=progress_fn,
                    mask=result.copy(),
                )
            res = run_one(lv)
            if res is not None:
                result[res] = 1

    return result


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


def skeletonize_slice(
    slice_: np.ndarray, algorithm: str = "skeletonize", **kwargs
) -> np.ndarray:
    """Skeletonize a single binary slice using the chosen algorithm."""
    img = slice_ > 0
    if algorithm == "skeletonize_3d":
        raise ValueError("skeletonize_3d requires a 3-D stack")
    if algorithm == "medial_axis":
        if medial_axis is not None:
            result = medial_axis(img, **kwargs)
            if isinstance(result, tuple):
                result = result[0]
            return result.astype(slice_.dtype)
    elif algorithm == "thin":
        try:
            from skimage.morphology import thin
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("thin import failed: %s", exc)
        else:
            return thin(img).astype(slice_.dtype)
    else:
        if skeletonize is not None:
            return skeletonize(img).astype(slice_.dtype)
    return _skeletonize_numpy(slice_)


def skeletonize_stack(
    stack: np.ndarray, algorithm: str = "skeletonize", **kwargs
) -> np.ndarray:
    """Skeletonize an entire mask stack or volume."""
    if algorithm == "skeletonize_3d":
        if _load_skeletonize_3d() is not None:
            return skeletonize_3d(stack > 0).astype(stack.dtype)
        # fallback: apply 2D skeletonization slice by slice
        return np.stack([_skeletonize_numpy(s) for s in stack])
    return np.stack([skeletonize_slice(s, algorithm, **kwargs) for s in stack])


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


def frangi_filter_slice(
    slice_: np.ndarray,
    sigmas: tuple[int | float, ...] = (1, 2, 3),
    *,
    black_ridges: bool = True,
) -> np.ndarray:
    """Enhance line structures using the Frangi vesselness filter."""
    if frangi is None:  # pragma: no cover - optional dependency
        return slice_.astype(float)
    return frangi(slice_.astype(float), sigmas=sigmas, black_ridges=black_ridges)


def sato_filter_slice(
    slice_: np.ndarray,
    sigmas: tuple[int | float, ...] = (1, 2, 3),
    *,
    black_ridges: bool = True,
) -> np.ndarray:
    """Enhance line structures using the Sato tubeness filter."""
    if sato is None:  # pragma: no cover - optional dependency
        return slice_.astype(float)
    return sato(slice_.astype(float), sigmas=sigmas, black_ridges=black_ridges)


def meijering_filter_slice(
    slice_: np.ndarray,
    sigmas: tuple[int | float, ...] = (1, 2, 3),
    *,
    black_ridges: bool = True,
) -> np.ndarray:
    """Enhance line structures using the Meijering neuriteness filter."""
    if meijering is None:  # pragma: no cover - optional dependency
        return slice_.astype(float)
    return meijering(slice_.astype(float), sigmas=sigmas, black_ridges=black_ridges)


def thin_slice(slice_: np.ndarray) -> np.ndarray:
    """Perform morphological thinning preserving topology."""
    try:
        from skimage.morphology import thin
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("thin import failed: %s", exc)
        return skeletonize_slice(slice_)
    return thin(slice_ > 0).astype(slice_.dtype)


def shortest_path_slice(
    image: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> np.ndarray:
    """Compute minimal cost path between ``start`` and ``end`` on ``image``."""
    try:
        from skimage.graph import route_through_array
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("graph import failed: %s", exc)
        return np.zeros_like(image, dtype=np.uint8)
    cost = 1.0 / (image.astype(float) + 1e-6)
    coords, _ = route_through_array(cost, start, end, fully_connected=True)
    result = np.zeros_like(image, dtype=np.uint8)
    for y, x in coords:
        result[int(y), int(x)] = 1
    return result


def ridge_filter_cv2_slice(slice_: np.ndarray) -> np.ndarray:
    """Detect ridges using OpenCV's RidgeDetectionFilter."""
    try:  # pragma: no cover - optional dependency
        import cv2  # type: ignore
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "RidgeDetectionFilter_create"):
            flt = cv2.ximgproc.RidgeDetectionFilter_create()
            return flt.getRidgeFilteredImage(slice_.astype(np.uint8)).astype(float)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("cv2 ridge filter failed: %s", exc)
    return slice_.astype(float)


def steger_ridge_slice(slice_: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Detect ridges using the ridge-detector package."""
    try:  # pragma: no cover - optional dependency
        import ridge_detector  # type: ignore
        detector = ridge_detector.RidgeDetector(sigma=sigma)
        return detector.detect_ridges(slice_.astype(float))
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("ridge_detector failed: %s", exc)
    return slice_.astype(float)


def chan_vese_slice(
    slice_: np.ndarray,
    iterations: int = 100,
    smoothing: int = 1,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    init_level_set: str | np.ndarray = "checkerboard",
) -> np.ndarray:
    """Segment ``slice_`` using morphological Chan-Vese evolution."""
    if morphological_chan_vese is None:  # pragma: no cover - optional dependency
        return np.zeros_like(slice_, dtype=np.uint8)
    result = morphological_chan_vese(
        slice_.astype(float),
        iterations,
        init_level_set=init_level_set,
        smoothing=smoothing,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    return result.astype(np.uint8)


def ced_filter_slice(
    slice_: np.ndarray,
    time_step: float = 0.125,
    conductance: float = 3.0,
    iterations: int = 5,
) -> np.ndarray:
    """Apply coherence-enhancing diffusion (ITK)."""
    try:  # pragma: no cover - optional dependency
        import itk  # type: ignore
        img = itk.image_view_from_array(slice_.astype(np.float32))
        flt = itk.CoherenceEnhancingDiffusionImageFilter.New(
            Input=img,
            TimeStep=time_step,
            ConductanceParameter=conductance,
            NumberOfIterations=iterations,
        )
        flt.Update()
        return itk.array_view_from_image(flt.GetOutput()).astype(float)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("itk CED failed: %s", exc)
    return slice_.astype(float)


def tubetk_segment_tubes_slice(slice_: np.ndarray) -> np.ndarray:
    """Automatic tube segmentation via TubeTK."""
    try:  # pragma: no cover - optional dependency
        import itk  # type: ignore
        if hasattr(itk, "tubetkSegmentTubes"):
            result = itk.tubetkSegmentTubes(slice_.astype(np.float32))
            if hasattr(result, "GetOutput"):
                result = result.GetOutput()
            return itk.array_view_from_image(result).astype(np.uint8)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("TubeTK segment tubes failed: %s", exc)
    return np.zeros_like(slice_, dtype=np.uint8)


def tubetk_path_grow_slice(slice_: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """Grow paths from seeds using TubeTK if available."""
    try:  # pragma: no cover - optional dependency
        import itk  # type: ignore
        if hasattr(itk, "tubetkExtractTubularGeodesicPaths"):
            img = itk.image_view_from_array(slice_.astype(np.float32))
            seed = itk.image_view_from_array(seed_mask.astype(np.uint8))
            flt = itk.tubetkExtractTubularGeodesicPaths.New(Input=img, SeedMask=seed)
            flt.Update()
            return itk.array_view_from_image(flt.GetOutput()).astype(np.uint8)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("TubeTK path grow failed: %s", exc)
    return seed_mask.astype(np.uint8)


def hessian_filter_slice(
    slice_: np.ndarray,
    sigmas: tuple[int | float, ...] = (1, 2, 3),
) -> np.ndarray:
    """Enhance ridges using the multiscale Hessian filter."""
    if hessian is None:  # pragma: no cover - optional dependency
        return slice_.astype(float)
    return hessian(slice_.astype(float), sigmas=sigmas)


def gabor_filter_slice(slice_: np.ndarray, frequency: float = 0.2) -> np.ndarray:
    """Apply the Gabor filter from scikit-image."""
    if gabor is None:  # pragma: no cover - optional dependency
        return slice_.astype(float)
    real, _ = gabor(slice_.astype(float), frequency=frequency)
    return real


def gabor_cv2_slice(
    slice_: np.ndarray,
    ksize: int = 21,
    sigma: float = 5.0,
    theta: float = 0.0,
    lambd: float = 10.0,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> np.ndarray:
    """Apply an OpenCV Gabor kernel via ``filter2D``."""
    try:  # pragma: no cover - optional dependency
        import cv2  # type: ignore
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        return cv2.filter2D(slice_.astype(np.float32), cv2.CV_32F, kernel)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("cv2 gabor filter failed: %s", exc)
    return slice_.astype(float)


def structure_tensor_eigen_slice(slice_: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues of the structure tensor."""
    if structure_tensor is None or structure_tensor_eigenvalues is None:  # pragma: no cover - optional dependency
        shape = slice_.shape
        return np.zeros(shape, dtype=float), np.zeros(shape, dtype=float)
    Axx, Axy, Ayy = structure_tensor(slice_.astype(float), sigma=sigma)
    A_elems = np.array([Axx, Axy, Ayy])
    eig = structure_tensor_eigenvalues(A_elems)
    return eig[0], eig[1]

