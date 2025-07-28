import xml.etree.ElementTree as ET
from typing import Any, Dict, List

__all__ = ["parse_zeiss_ome_metadata", "format_metadata"]

def _get_text(elem: ET.Element | None, default: str | None = None) -> str | None:
    if elem is None:
        return default
    return elem.text or default

def parse_zeiss_ome_metadata(xml: str) -> Dict[str, Any]:
    """Parse Zeiss-style OME metadata into a dictionary."""
    root = ET.fromstring(xml)
    info: Dict[str, Any] = {}

    # ---- document / user info ----
    doc = root.find(".//Document")
    if doc is not None:
        info["document"] = {
            "name": _get_text(doc.find("Name")),
            "title": _get_text(doc.find("Title")),
            "description": _get_text(doc.find("Description")),
            "user": _get_text(doc.find("UserName")),
            "creation_date": _get_text(doc.find("CreationDate")),
        }
    user_display = root.findtext(".//User/DisplayName")
    if user_display:
        info.setdefault("document", {})["display_name"] = user_display

    # ---- image dimensions ----
    image = root.find(".//Image")
    if image is not None:
        dims = image.find("Dimensions")
        img_info = {
            "pixel_type": _get_text(image.find("PixelType")),
            "component_bits": _get_text(image.find("ComponentBitCount")),
        }
        if dims is not None:
            for key in ("SizeX", "SizeY", "SizeZ", "SizeM"):
                val = _get_text(dims.find(key))
                if val is not None:
                    img_info[key.lower()] = int(val)
        info["image"] = img_info

    # ---- scaling information ----
    scaling = root.find(".//Scaling/Items")
    if scaling is not None:
        scale_info: Dict[str, float] = {}
        for dist in scaling.findall("Distance"):
            sid = dist.attrib.get("Id")
            if sid:
                try:
                    scale_info[sid] = float(dist.findtext("Value", "0"))
                except Exception:
                    pass
        if scale_info:
            info["scaling"] = scale_info

    # ---- channels ----
    channels: List[Dict[str, Any]] = []
    for ch in root.findall(".//Channels/Channel"):
        channels.append(
            {
                "id": ch.attrib.get("Id"),
                "name": ch.attrib.get("Name"),
                "illumination": _get_text(ch.find("IlluminationType")),
                "excitation": _get_text(ch.find("ExcitationWavelength")),
                "emission": _get_text(ch.find("EmissionWavelength")),
            }
        )
    if channels:
        info["channels"] = channels

    # ---- instrument ----
    instrument = root.find(".//Instrument")
    if instrument is not None:
        microscopes = [
            _get_text(m.find("System")) for m in instrument.findall("./Microscopes/Microscope")
        ]
        objectives = [
            _get_text(o.find("./Manufacturer/Model"))
            for o in instrument.findall(".//Objectives/Objective")
        ]
        light_sources = [
            _get_text(ls.find("./Manufacturer/Model"))
            for ls in instrument.findall(".//LightSource")
        ]
        info["instrument"] = {
            "microscopes": [m for m in microscopes if m],
            "objectives": [o for o in objectives if o],
            "light_sources": [ls for ls in light_sources if ls],
        }

    # ---- custom attributes (environment etc.) ----
    custom: Dict[str, str] = {}
    for tag in root.findall(".//CustomAttributes/LsmTag"):
        name = tag.attrib.get("Name")
        val = tag.text
        if name and val is not None:
            custom[name] = val
    if custom:
        info["custom_attributes"] = custom

    # ---- acquisition parameters ----
    acq = root.find(".//AcquisitionModeSetup")
    if acq is not None:
        acq_info: Dict[str, Any] = {}
        for key in (
            "AcquisitionMode",
            "DimensionX",
            "DimensionY",
            "DimensionZ",
            "DimensionT",
        ):
            val = _get_text(acq.find(key))
            if val is not None:
                acq_info[key.lower()] = val
        if acq_info:
            info["acquisition"] = acq_info

    # ---- display settings ----
    disp_chans: List[Dict[str, Any]] = []
    for ch in root.findall(".//DisplaySetting/Channels/Channel"):
        disp_chans.append(
            {
                "name": ch.attrib.get("Name"),
                "color": _get_text(ch.find("Color")),
                "low": _get_text(ch.find("Low")),
                "high": _get_text(ch.find("High")),
                "gamma": _get_text(ch.find("Gamma")),
            }
        )
    if disp_chans:
        info["display"] = disp_chans

    return info


def format_metadata(info: Dict[str, Any]) -> str:
    """Return a short formatted summary of parsed metadata."""
    lines: List[str] = []
    doc = info.get("document", {})
    if doc:
        name = doc.get("name")
        if name:
            lines.append(f"Name: {name}")
        if doc.get("display_name"):
            lines.append(f"User: {doc['display_name']}")
        elif doc.get("user"):
            lines.append(f"User: {doc['user']}")
        if doc.get("creation_date"):
            lines.append(f"Date: {doc['creation_date']}")

    img = info.get("image", {})
    if img:
        if all(k in img for k in ("sizex", "sizey", "sizez")):
            lines.append(
                f"Dimensions: {img['sizex']} x {img['sizey']} x {img['sizez']}"
            )
        if img.get("pixel_type"):
            lines.append(f"Pixel type: {img['pixel_type']}")

    scaling = info.get("scaling")
    if isinstance(scaling, dict) and scaling:
        x = scaling.get("X")
        y = scaling.get("Y")
        z = scaling.get("Z")
        if x is not None and y is not None and z is not None:
            lines.append(f"Pixel size: {x} x {y} x {z}")

    channels = info.get("channels")
    if channels:
        ch_names = ", ".join(ch.get("name", "") for ch in channels)
        lines.append(f"Channels: {ch_names}")

    instr = info.get("instrument", {})
    if instr.get("microscopes"):
        lines.append(
            "Microscope: " + ", ".join(instr.get("microscopes", []))
        )

    return "\n".join(lines)
