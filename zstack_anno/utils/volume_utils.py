from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


KNOWN_2P5D_DIRNAMES = (
    "2026-02-05_7ch_combo_pred_zstacks",
    "7chcombonnunet_inference",
)
THREE_D_BUNDLE_GLOB = "*3d_vanilla*dataset007*chunked"


@dataclass(frozen=True)
class VolumeSource:
    """A render-ready 3-D volume with physical spacing metadata."""

    label: str
    volume_zyx: np.ndarray
    spacing_xyz: tuple[float, float, float]
    path: str | None = None
    kind: str = "raw"
    note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.volume_zyx.shape)

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        sx, sy, sz = self.spacing_xyz
        return (sz, sy, sx)


def load_volume_file(path: str) -> tuple[np.ndarray, str | None]:
    """Load a TIFF/OME-TIFF volume and normalize it to ZYX."""
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
        ome_metadata = tif.ome_metadata

    arr = np.squeeze(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape={arr.shape}")
    return arr, ome_metadata


def build_volume_source_from_file(
    path: str,
    *,
    label: str | None = None,
    kind: str = "raw",
    spacing_override_xyz: tuple[float, float, float] | None = None,
    note: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> VolumeSource:
    arr, ome_metadata = load_volume_file(path)
    spacing_xyz = spacing_override_xyz or extract_ome_spacing_xyz(ome_metadata) or (1.0, 1.0, 1.0)
    meta = dict(metadata or {})
    if ome_metadata:
        meta.setdefault("ome_metadata", ome_metadata)
    return VolumeSource(
        label=label or Path(path).name,
        volume_zyx=np.asarray(arr),
        spacing_xyz=tuple(float(v) for v in spacing_xyz),
        path=path,
        kind=kind,
        note=note,
        metadata=meta,
    )


def infer_spacing_xyz_from_reference(
    reference_shape_zyx: tuple[int, int, int],
    reference_spacing_xyz: tuple[float, float, float],
    target_shape_zyx: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Infer spacing for a target grid that should cover the same physical extent."""
    ref_z, ref_y, ref_x = (float(v) for v in reference_shape_zyx)
    tgt_z, tgt_y, tgt_x = (float(v) for v in target_shape_zyx)
    sx, sy, sz = (float(v) for v in reference_spacing_xyz)
    if min(tgt_z, tgt_y, tgt_x) <= 0:
        raise ValueError(f"Invalid target shape: {target_shape_zyx}")
    return (
        sx * (ref_x / tgt_x),
        sy * (ref_y / tgt_y),
        sz * (ref_z / tgt_z),
    )


def build_mask_source_with_reference_extent(
    mask_path: str,
    *,
    reference_shape_zyx: tuple[int, int, int],
    reference_spacing_xyz: tuple[float, float, float],
    label: str | None = None,
    note: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> VolumeSource:
    """Build a mask source, inferring spacing from a reference raw stack when needed."""
    arr, ome_metadata = load_volume_file(mask_path)
    inferred = False
    spacing_xyz = extract_ome_spacing_xyz(ome_metadata)
    if spacing_xyz is None:
        spacing_xyz = infer_spacing_xyz_from_reference(
            tuple(int(v) for v in reference_shape_zyx),
            tuple(float(v) for v in reference_spacing_xyz),
            tuple(int(v) for v in arr.shape),
        )
        inferred = True
    meta = dict(metadata or {})
    if ome_metadata:
        meta.setdefault("ome_metadata", ome_metadata)
    if inferred:
        meta.setdefault("spacing_inferred_from_reference", True)
        meta.setdefault("reference_shape_zyx", tuple(int(v) for v in reference_shape_zyx))
        meta.setdefault("reference_spacing_xyz", tuple(float(v) for v in reference_spacing_xyz))
        inferred_note = (
            "Physical spacing inferred from current raw stack extent because prediction metadata "
            "did not include OME voxel sizes."
        )
        note = f"{note} {inferred_note}".strip() if note else inferred_note
    return VolumeSource(
        label=label or Path(mask_path).name,
        volume_zyx=np.asarray(arr),
        spacing_xyz=tuple(float(v) for v in spacing_xyz),
        path=mask_path,
        kind="mask",
        note=note,
        metadata=meta,
    )


def normalize_stack_id(name_or_path: str) -> str:
    """Normalize stack file names across raw, 2.5D pred, and resampled 3D raw."""
    name = Path(name_or_path).name
    suffixes = (
        ".ome.tiff",
        ".ome.tif",
        ".tiff",
        ".tif",
        ".ome",
        ".pred",
        ".mask",
        ".seg",
    )
    changed = True
    while changed:
        changed = False
        lowered = name.lower()
        for suffix in suffixes:
            if lowered.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break
    name = re.sub(r"(?:\.ome)?_dz\d+(?:p\d+)?$", "", name, flags=re.IGNORECASE)
    return name


def extract_ome_spacing_xyz(ome_xml: str | None) -> tuple[float, float, float] | None:
    """Return physical spacing in OME order (X, Y, Z)."""
    if not ome_xml:
        return None
    try:
        root = ET.fromstring(ome_xml)
    except Exception:
        return None
    pixels = _find_pixels_element(root)
    if pixels is None:
        return None
    x_raw = pixels.attrib.get("PhysicalSizeX")
    y_raw = pixels.attrib.get("PhysicalSizeY")
    z_raw = pixels.attrib.get("PhysicalSizeZ")
    if x_raw is None or y_raw is None or z_raw is None:
        return None
    try:
        x = float(x_raw)
        y = float(y_raw)
        z = float(z_raw)
    except Exception:
        return None
    return (x, y, z)


def _find_pixels_element(root: ET.Element) -> ET.Element | None:
    for elem in root.iter():
        tag = elem.tag
        if isinstance(tag, str) and tag.rsplit("}", 1)[-1] == "Pixels":
            return elem
    return None


def find_confocal_data_root(path: str) -> Path | None:
    """Find the 'Confocal Myelin data' root for a given stack path."""
    current = Path(path).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == "Confocal Myelin data":
            return parent
    return None


def read_json_file(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def find_matching_confocal_inference(
    raw_path: str,
    *,
    inference_root: str | None = None,
) -> dict[str, Any]:
    """Locate sibling inference assets for a raw confocal tile stack."""
    stack_id = normalize_stack_id(raw_path)
    root = Path(inference_root) if inference_root else None
    if root is None:
        confocal_root = find_confocal_data_root(raw_path)
        if confocal_root is not None:
            root = confocal_root / "Inference"

    result: dict[str, Any] = {
        "stack_id": stack_id,
        "inference_root": str(root) if root else None,
        "two_p_five_d_mask_path": None,
        "three_d_resampled_raw_path": None,
        "three_d_meta_json_path": None,
        "three_d_mask_path": None,
        "three_d_bundle_root": None,
    }
    if root is None or not root.exists():
        return result

    two_p_dirs: list[Path] = []
    for dirname in KNOWN_2P5D_DIRNAMES:
        path = root / dirname
        if path.exists():
            two_p_dirs.append(path)
    for path in sorted(root.glob("*7ch*pred*zstacks*")):
        if path.is_dir() and path not in two_p_dirs:
            two_p_dirs.append(path)

    for two_p_dir in two_p_dirs:
        matches = sorted(two_p_dir.glob(f"{stack_id}*.pred*.tif*"))
        if matches:
            result["two_p_five_d_mask_path"] = str(matches[0])
            break

    for bundle in sorted(root.glob(THREE_D_BUNDLE_GLOB)):
        raw_matches = sorted((bundle / "original_zstacks").glob(f"{stack_id}*.tif*"))
        meta_matches = sorted((bundle / "meta").glob(f"**/{stack_id}.ome.json"))
        pred_matches = [
            path for path in sorted(bundle.glob(f"**/{stack_id}*pred*.tif*"))
            if "original_zstacks" not in path.parts
        ]
        if not raw_matches and not meta_matches and not pred_matches:
            continue
        result["three_d_bundle_root"] = str(bundle)
        if raw_matches:
            result["three_d_resampled_raw_path"] = str(raw_matches[0])
        if meta_matches:
            result["three_d_meta_json_path"] = str(meta_matches[0])
        if pred_matches:
            result["three_d_mask_path"] = str(pred_matches[0])
        break

    return result


def build_resampled_three_d_raw_source(
    raw_path: str,
    resampled_raw_path: str,
    meta_json_path: str | None,
) -> VolumeSource:
    """Build a 3-D-ready raw source with inferred target Z spacing."""
    _, raw_ome = load_volume_file(raw_path)
    raw_spacing = extract_ome_spacing_xyz(raw_ome) or (1.0, 1.0, 1.0)
    metadata: dict[str, Any] = {}
    spacing_xyz = raw_spacing
    note = None
    if meta_json_path and Path(meta_json_path).exists():
        metadata = read_json_file(meta_json_path)
        dz_target = metadata.get("dz_target")
        if isinstance(dz_target, (int, float)):
            spacing_xyz = (raw_spacing[0], raw_spacing[1], float(dz_target))
            note = (
                f"Resampled raw for 3D inference, Z spacing {raw_spacing[2]:.4f}"
                f" -> {float(dz_target):.4f}"
            )
    return build_volume_source_from_file(
        resampled_raw_path,
        label="3D aligned raw",
        kind="raw",
        spacing_override_xyz=spacing_xyz,
        note=note,
        metadata=metadata,
    )
