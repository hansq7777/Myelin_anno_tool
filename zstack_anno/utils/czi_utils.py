"""Utilities for handling CZI files."""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import os
import numpy as np
import tifffile

try:
    import czifile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    czifile = None  # type: ignore


class CziNotSupportedError(RuntimeError):
    """Raised when CZI support is unavailable."""


def read_czi_metadata(path: str) -> Dict[str, object]:
    """Read a CZI file and return parsed metadata information."""
    if czifile is None:
        raise CziNotSupportedError(
            "CZI support requires the 'czifile' package to be installed"
        )

    with czifile.CziFile(path) as czi:
        metadata = czi.metadata()

    return _parse_czi_metadata(metadata)

def _parse_czi_metadata(metadata: str) -> Dict[str, object]:
    """Parse relevant information from CZI XML metadata.

    Parameters
    ----------
    metadata:
        XML metadata string returned by ``CziFile.metadata()``.

    Returns
    -------
    dict
        Dictionary containing ``stage_positions`` (list of ``(x, y)``),
        ``stack_count`` (int) and ``pixel_size`` (``(x, y, z)`` or ``None``).
    """
    import xml.etree.ElementTree as ET

    stage_positions: List[Tuple[float, float]] = []
    pixel_size: Optional[Tuple[float, float, float]] = None

    try:
        root = ET.fromstring(metadata)

        # collect stage positions
        for pos in root.findall('.//StagePosition'):
            try:
                idx = int(pos.attrib.get('Index', len(stage_positions)))
            except Exception:
                idx = len(stage_positions)
            x = float(pos.attrib.get('X', '0'))
            y = float(pos.attrib.get('Y', '0'))
            while len(stage_positions) <= idx:
                stage_positions.append((0.0, 0.0))
            stage_positions[idx] = (x, y)

        # fall back to Scene elements if present
        if not stage_positions:
            for idx, scene in enumerate(root.findall('.//Scene')):
                x = float(scene.attrib.get('CenterX', '0'))
                y = float(scene.attrib.get('CenterY', '0'))
                stage_positions.append((x, y))

        # pixel size information
        scaling = root.find('.//Scaling')
        if scaling is not None:
            try:
                px_x = float(scaling.attrib.get('X', '1'))
                px_y = float(scaling.attrib.get('Y', '1'))
                px_z = float(scaling.attrib.get('Z', '1'))
                pixel_size = (px_x, px_y, px_z)
            except Exception:
                pass

        if pixel_size is None:
            pixels = root.find('.//Pixels')
            if pixels is not None:
                try:
                    px_x = float(pixels.attrib.get('PhysicalSizeX', '1'))
                    px_y = float(pixels.attrib.get('PhysicalSizeY', '1'))
                    px_z = float(pixels.attrib.get('PhysicalSizeZ', '1'))
                    pixel_size = (px_x, px_y, px_z)
                except Exception:
                    pixel_size = None

    except Exception:
        pass

    return {
        'stage_positions': stage_positions,
        'stack_count': len(stage_positions),
        'pixel_size': pixel_size,
    }

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

    info = _parse_czi_metadata(metadata)
    positions = info.get('stage_positions', [])

    written: List[str] = []
    base = os.path.splitext(os.path.basename(path))[0]

    for idx, stack in enumerate(arr):
        # squeeze possible singleton dimensions
        stack = np.squeeze(stack)
        stage_x = 0.0
        stage_y = 0.0
        if idx < len(positions):
            stage_x, stage_y = positions[idx]

        name = f"{base}_X{stage_x:.1f}_Y{stage_y:.1f}.ome.tif"
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
