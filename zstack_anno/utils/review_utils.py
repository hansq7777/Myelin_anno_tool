from __future__ import annotations

import os

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
        drive = text[0].lower()
        tail = text[3:].replace("\\", "/")
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
