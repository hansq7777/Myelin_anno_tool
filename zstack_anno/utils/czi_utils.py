"""Utilities for handling CZI files."""
from __future__ import annotations

from typing import List
import os
import numpy as np
import tifffile

try:
    import czifile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    czifile = None  # type: ignore


class CziNotSupportedError(RuntimeError):
    """Raised when CZI support is unavailable."""


def split_czi_file(path: str, out_dir: str) -> List[str]:
    """Split a CZI file into individual Z stacks saved as OME-TIFF.

    Parameters
    ----------
    path:
        Path to the ``.czi`` file.
    out_dir:
        Directory where the extracted stacks will be saved.

    Returns
    -------
    list[str]
        List of written file paths.
    """
    if czifile is None:
        raise CziNotSupportedError(
            "CZI support requires the 'czifile' package to be installed"
        )

    with czifile.CziFile(path) as czi:
        metadata = czi.metadata()
        # ``.asarray`` returns data ordered as (S, T, C, Z, Y, X)
        arr = czi.asarray()

    if arr.ndim < 5:
        arr = arr[np.newaxis]

    written: List[str] = []

    for idx, stack in enumerate(arr):
        # squeeze possible singleton dimensions
        stack = np.squeeze(stack)
        stage_x = 0.0
        stage_y = 0.0
        # attempt to parse stage position from metadata
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(metadata)
            pos = root.find(f".//StagePosition[@Index='{idx}']")
            if pos is not None:
                stage_x = float(pos.attrib.get("X", "0"))
                stage_y = float(pos.attrib.get("Y", "0"))
        except Exception:
            pass

        name = f"stack_X{stage_x:.1f}_Y{stage_y:.1f}.ome.tif"
        out_path = os.path.join(out_dir, name)
        tifffile.imwrite(out_path, stack, ome=metadata)
        written.append(out_path)

    return written


def czi_to_tiff(path: str, out_path: str) -> str:
    """Save the entire CZI image as a single OME-TIFF stack.

    Parameters
    ----------
    path:
        Path to the ``.czi`` file.
    out_path:
        Output file path for the OME-TIFF stack.

    Returns
    -------
    str
        The written file path.
    """
    if czifile is None:
        raise CziNotSupportedError(
            "CZI support requires the 'czifile' package to be installed"
        )

    with czifile.CziFile(path) as czi:
        metadata = czi.metadata()
        arr = czi.asarray()

    arr = np.squeeze(arr)
    if arr.ndim > 3:
        # collapse all leading dimensions except Y and X
        leading = int(np.prod(arr.shape[:-2]))
        arr = arr.reshape(leading, arr.shape[-2], arr.shape[-1])

    tifffile.imwrite(out_path, arr, ome=metadata)
    return out_path
