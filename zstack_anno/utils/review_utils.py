from __future__ import annotations

import os


def _parse_windows_drive_map() -> dict[str, str]:
    """Parse ``ZSTACK_WINDOWS_DRIVE_MAP`` into ``{drive_letter: posix_root}``.

    Example:
        ``ZSTACK_WINDOWS_DRIVE_MAP="D=/data/myelin;E=/mnt/extra"``
    """
    raw = os.getenv("ZSTACK_WINDOWS_DRIVE_MAP", "")
    if not raw:
        return {}

    mapping: dict[str, str] = {}
    for part in raw.split(";"):
        item = part.strip()
        if not item or "=" not in item:
            continue
        drive, root = item.split("=", 1)
        drive = drive.strip().upper().rstrip(":")
        root = root.strip().replace("\\", "/").rstrip("/")
        if len(drive) == 1 and root:
            mapping[drive] = root
    return mapping


def _join_posix(root: str, tail: str) -> str:
    root = root.rstrip("/")
    tail = (tail or "").replace("\\", "/").lstrip("/")
    if not tail:
        return root
    return f"{root}/{tail}"


def normalize_review_grade(value: object) -> str:
    """Return normalized review grade: '', 'A', 'B' or 'C'."""
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text in {"A", "B", "C"}:
        return text
    return ""


def windows_to_local_path(path: str) -> str:
    """Normalize tracker path for current OS.

    - On POSIX/WSL: convert ``D:\\foo\\bar`` -> ``/mnt/d/foo/bar``
      (or use ``ZSTACK_WINDOWS_DRIVE_MAP`` for native Linux roots)
    - On Windows: keep drive paths as-is; convert ``/mnt/d/foo/bar`` -> ``D:\\foo\\bar``
    """
    text = (path or "").strip().strip('"')
    if not text:
        return ""
    if os.name == "nt":
        if text.startswith("/mnt/") and len(text) > 7 and text[6] == "/":
            drive = text[5].upper()
            tail = text[7:].replace("/", "\\")
            return f"{drive}:\\{tail}"
        return text.replace("/", "\\")
    if text.startswith("/mnt/"):
        return text
    # Drive path, e.g. C:\data\file.tif
    if len(text) >= 3 and text[1] == ":" and text[2] in ("\\", "/"):
        drive_upper = text[0].upper()
        tail = text[3:].replace("\\", "/")
        mapped_root = _parse_windows_drive_map().get(drive_upper)
        if mapped_root:
            return _join_posix(mapped_root, tail)
        drive = drive_upper.lower()
        return f"/mnt/{drive}/{tail}"
    return text.replace("\\", "/")


def local_to_windows_path(path: str) -> str:
    """Convert path to Windows style path for tracker persistence."""
    text = (path or "").strip()
    if not text:
        return ""
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5].upper()
        tail = text[7:].replace("/", "\\")
        return f"{drive}:\\{tail}"
    normalized = text.replace("\\", "/")
    for drive, root in _parse_windows_drive_map().items():
        root_norm = root.replace("\\", "/").rstrip("/")
        if normalized == root_norm:
            return f"{drive}:\\"
        if normalized.startswith(root_norm + "/"):
            tail = normalized[len(root_norm) + 1 :].replace("/", "\\")
            return f"{drive}:\\{tail}"
    if os.name == "nt":
        return text.replace("/", "\\")
    return text


def build_inference_name_candidates(zstack_id: str) -> list[str]:
    """Return accepted inference filename candidates for a given zstack id."""
    zid = (zstack_id or "").strip()
    if not zid:
        return []
    candidates: list[str] = []
    # Common naming: 2501_60_R_IL_S00.pred.ome.tif
    if zid.endswith(".ome"):
        candidates.append(f"{zid[:-4]}.pred.ome.tif")
    else:
        candidates.append(f"{zid}.pred.ome.tif")
    # Alternate naming used in some batches: 2501_60_R_IL_S00.ome.pred.ome.tif
    candidates.append(f"{zid}.pred.ome.tif")
    # Keep insertion order while removing duplicates
    deduped: list[str] = []
    for name in candidates:
        if name and name not in deduped:
            deduped.append(name)
    return deduped
