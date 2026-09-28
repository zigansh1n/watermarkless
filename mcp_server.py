"""MCP server for watermarkless (stdio transport).

    python mcp_server.py

Tools work on local files so an agent can point at a downloaded listing
folder. Cleaned copies go to `<dir>/clean/<name>.jpg` unless an output path
is given; originals are never modified.
"""

import sys
import time
from pathlib import Path
from typing import Optional

try:  # mcp 2.x renamed FastMCP to MCPServer; keep 1.x working too
    from mcp.server.mcpserver import MCPServer as FastMCP
except ModuleNotFoundError:
    from mcp.server.fastmcp import FastMCP
from PIL import Image

import app

server = FastMCP(
    "watermarkless",
    instructions=(
        "Removes watermarks from photos on this machine. Known marks are profiles "
        "(list_profiles; 'avito' is built in): detect_watermark checks a photo, clean_photo "
        "cleans one file, clean_folder a downloaded listing. profile='auto' finds marks of "
        "any site with a generic detector (find_watermarks shows the boxes); mask_box="
        "[x0,y0,x1,y1] inpaints exactly that area without a check. "
        "Only process your own photos or photos you have permission to use."
    ),
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _open(path: str) -> Image.Image:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"no such file: {source}")
    if max(Image.open(source).size) > app.MAX_SIDE:
        raise ValueError(f"image side exceeds {app.MAX_SIDE}px")
    return Image.open(source).convert("RGB")


def _default_output(source: Path) -> Path:
    return source.parent / "clean" / (source.stem + ".jpg")


def _clean_one(source: Path, output: Optional[Path], method: str, quality: int, profile: str = "avito", mask_box: Optional[list] = None) -> dict:
    image = _open(str(source))
    started = time.time()
    if mask_box is not None:
        if len(mask_box) != 4:
            raise ValueError("mask_box must be [x0, y0, x1, y1]")
        cleaned = app.cleaner.clean_mask(image, app.box_mask(image.width, image.height, tuple(int(v) for v in mask_box)))
        target = output or _default_output(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        cleaned.save(target, format="JPEG", quality=quality, subsampling=0)
        return {"path": str(source), "status": "cleaned", "output": str(target), "method": "lama", "maskBox": list(mask_box), "ms": round((time.time() - started) * 1000)}
    if profile.strip().lower() == app.AUTO:
        cleaned, boxes = app.cleaner.clean_auto(image)
        if cleaned is None:
            return {"path": str(source), "status": "no_watermark", "profile": app.AUTO}
        target = output or _default_output(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        cleaned.save(target, format="JPEG", quality=quality, subsampling=0)
        return {"path": str(source), "status": "cleaned", "profile": app.AUTO, "output": str(target), "method": "auto+lama", "boxes": [list(b) for b in boxes], "ms": round((time.time() - started) * 1000)}
    chosen = app.profile_named(profile)
    before = app.presence_score(image, chosen)
    if before is not None and before < chosen.found:
        return {"path": str(source), "status": "no_watermark", "profile": chosen.name, "score": round(before)}
    cleaned, used = app.cleaner.clean(image, "logo", method, chosen)
    after = app.presence_score(cleaned, chosen)
    if after is not None and after > chosen.clean:
        return {"path": str(source), "status": "still_present", "profile": chosen.name, "scoreBefore": round(before or 0), "scoreAfter": round(after), "method": used}
    target = output or _default_output(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(target, format="JPEG", quality=quality, subsampling=0)
    return {
        "path": str(source),
        "status": "cleaned",
        "profile": chosen.name,
        "output": str(target),
        "method": used,
        "scoreBefore": None if before is None else round(before),
        "scoreAfter": None if after is None else round(after),
        "ms": round((time.time() - started) * 1000),
    }


@server.tool()
def cleaner_health() -> dict:
    """Model, detector and device state of the cleaner."""
    return {
        "model": "lama",
        "device": app.cleaner.device,
        "detector": app.detector.available,
        "unblend": app.cleaner.unblender.available,
        "requests": app.cleaner.requests,
    }


@server.tool()
def list_profiles() -> dict:
    """Known watermark profiles with their block geometry, detector availability and thresholds, plus whether the generic 'auto' detector is available."""
    return {"profiles": [app.profile_info(p) for p in app.PROFILES.values()], "auto": app.cleaner.auto.available}


@server.tool()
def find_watermarks(path: str, confidence: float = 0.25) -> dict:
    """Locate watermarks or logos of any site on a photo with the generic YOLO detector. Returns boxes [x0, y0, x1, y1, confidence] in image pixels; pass one to clean_photo as mask_box (add ~10 px) or just use profile='auto'."""
    try:
        boxes = app.cleaner.auto.boxes(_open(path), conf=confidence)
    except ValueError as error:
        return {"path": path, "status": "error", "error": str(error)}
    return {"path": path, "boxes": [list(b) for b in boxes]}


@server.tool()
def detect_watermark(path: str, profile: str = "avito") -> dict:
    """Score how strongly a photo carries a profile's watermark (not for 'auto'; use find_watermarks). For avito: present at or above 400, clean at or below 250."""
    try:
        chosen = app.profile_named(profile)
        score = app.presence_score(_open(path), chosen)
    except ValueError as error:
        return {"path": path, "status": "error", "error": str(error)}
    return {
        "path": path,
        "profile": chosen.name,
        "score": None if score is None else round(score),
        "present": None if score is None else score >= chosen.found,
        "thresholds": {"present": chosen.found, "clean": chosen.clean},
    }


@server.tool()
def clean_photo(
    path: str,
    output_path: Optional[str] = None,
    method: str = "lama",
    quality: int = 95,
    profile: str = "avito",
    mask_box: Optional[list] = None,
) -> dict:
    """Remove a watermark from one photo. Writes <dir>/clean/<name>.jpg unless output_path is given. profile: a known mark (avito) with its mask and detector, or 'auto' to find marks of any site with the generic detector; mask_box=[x0,y0,x1,y1] in pixels inpaints exactly that area without a check. method: lama (default) or unblend (experimental, avito only)."""
    if method not in ("lama", "unblend"):
        raise ValueError("method must be lama or unblend")
    if not 1 <= quality <= 100:
        raise ValueError("quality must be 1..100")
    source = Path(path).expanduser()
    try:
        return _clean_one(source, Path(output_path).expanduser() if output_path else None, method, quality, profile, mask_box)
    except ValueError as error:
        return {"path": path, "status": "error", "error": str(error)}


@server.tool()
def clean_folder(
    directory: str,
    method: str = "lama",
    quality: int = 95,
    skip_existing: bool = True,
    profile: str = "avito",
    mask_box: Optional[list] = None,
) -> dict:
    """Remove a watermark from every photo in a folder (not recursive; a `clean` subfolder is ignored), by profile ('auto' for any site) or by a shared mask_box. Returns per-file results."""
    if method not in ("lama", "unblend"):
        raise ValueError("method must be lama or unblend")
    if profile.strip().lower() != app.AUTO:
        app.profile_named(profile)
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError(f"no such directory: {root}")
    results = []
    for source in sorted(root.iterdir()):
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        target = _default_output(source)
        if skip_existing and target.is_file():
            results.append({"path": str(source), "status": "skipped", "output": str(target)})
            continue
        try:
            results.append(_clean_one(source, None, method, quality, profile, mask_box))
        except Exception as error:  # noqa: BLE001 - one bad file must not stop the folder
            results.append({"path": str(source), "status": "error", "error": str(error)})
    counts = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"directory": str(root), "counts": counts, "results": results}


if __name__ == "__main__":
    server.run(transport="stdio")
    sys.exit(0)
