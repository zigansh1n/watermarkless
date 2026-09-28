"""watermarkless: removes watermarks from photos with LaMa inpainting.

Three ways to say where the mark is:

* a *profile* — one known watermark at a fixed place (`profiles/<name>/`):
  block position, mask silhouette, a presence detector and optional overlay
  maps. `avito` is built in; `fit_profile.py` makes new ones from samples.
* `auto` — a generic YOLO watermark detector (corzent/yolo11x_watermark_detection,
  MIT) finds marks anywhere on the photo; each box, padded, is inpainted.
  Slower and less exact than a profile, needs no setup.
* a caller-supplied `mask_box` — exactly that rectangle, no detection.

Default method is LaMa inpainting of the mask. `method=unblend` inverts a
profile's fitted overlay `obs = (1-a)*orig + a*c` (`fit_alpha.py`) and hands
only near-opaque pixels to LaMa; it is experimental and never chosen
automatically.

HTTP API
  GET  /health
  GET  /v1/profiles
  POST /v1/clean                -> image/jpeg
       body: raw image bytes (Content-Type image/*) or multipart field `file`
       query: profile=avito (default) | profile=auto | mask_box=x0,y0,x1,y1
              mask=logo|rect (profile silhouette or its whole block)
              force=1 (clean the profile mask even when the detector says absent)
              quality=1..100 (default 95), method=auto|lama|unblend (auto = lama)
       auth: Authorization: Bearer <CLEANER_TOKEN>; required unless CLEANER_ALLOW_ANON=1
       reply headers: X-Mask-Box, X-Watermark-Before, X-Watermark-After, X-Method
       422 when a profile watermark is still detectable after cleaning
       204 when nothing is detected (image left unchanged)

Only a window around the mask is processed and only the masked pixels are
written back, so the rest of the photo changes only by the JPEG re-encode.
"""

import hashlib
import io
import json
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
PROFILES_DIR = HERE / "profiles"

MASK_DILATE = 4
# Context LaMa sees around the mask; enough for wood grain and tile lines.
WINDOW_PAD = 192
MAX_SIDE = 4096
MAX_BODY = 25 * 1024 * 1024
# Above this alpha the inverse amplifies JPEG noise more than it restores.
LAMA_ALPHA_THRESHOLD = 0.6
DETECTOR_PAD = 8

# Generic detector: fine-tuned YOLO11x for watermarks and logos, MIT licence.
YOLO_URL = "https://huggingface.co/corzent/yolo11x_watermark_detection/resolve/main/best.pt"
YOLO_SHA256 = "6ac71b6ab8db27ec7928b5176e60a359c65e1579a5c1d58cf2f98df30cf3085e"
YOLO_PATH = Path(os.environ.get("WATERMARKLESS_YOLO", HERE / "weights" / "yolo11x_watermark.pt"))
YOLO_CONF = 0.25
YOLO_IMGSZ = 1024
# Detector boxes clip glyph edges by a few pixels; grow them before masking.
AUTO_PAD = 10
AUTO = "auto"


@dataclass(frozen=True)
class Profile:
    """One known watermark at a fixed offset from the bottom-right corner.

    `width`, `height`, `right`, `bottom` are pixels on a photo `ref_width`
    wide. Two ways the geometry follows the photo size: `scaled` multiplies
    everything by width/ref_width; `right_ratio`/`bottom_ratio` add a share
    of the width/height to the margins while the block keeps its pixel size
    (Cian: a ~205x90 px mark 10% of the width from the right and 8% of the
    height from the bottom). Avito uses neither: the same 103x37 block sits
    6/5 px from the corner on every variant."""

    name: str
    width: int
    height: int
    right: int
    bottom: int
    directory: Path
    found: float
    clean: float
    notes: str = ""
    scaled: bool = False
    ref_width: int = 1280
    right_ratio: float = 0.0
    bottom_ratio: float = 0.0

    @property
    def template(self) -> Path:
        return self.directory / "mask.png"

    @property
    def detector(self) -> Path:
        return self.directory / "detector.png"

    @property
    def detector_meta(self) -> Path:
        return self.directory / "detector.json"

    @property
    def k_map(self) -> Path:
        return self.directory / "k.png"

    @property
    def b_map(self) -> Path:
        return self.directory / "b.png"

    def scale(self, image_w: int) -> float:
        return image_w / self.ref_width if self.scaled else 1.0

    def block_size(self, image_w: int) -> tuple:
        k = self.scale(image_w)
        return (max(1, round(self.width * k)), max(1, round(self.height * k)))

    def box(self, image_w: int, image_h: int) -> tuple:
        """Bounding box (x0, y0, x1, y1) of the watermark block on an image."""
        k = self.scale(image_w)
        w, h = self.block_size(image_w)
        x1 = min(image_w, image_w - round(self.right * k + self.right_ratio * image_w))
        y1 = min(image_h, image_h - round(self.bottom * k + self.bottom_ratio * image_h))
        return (max(0, x1 - w), max(0, y1 - h), x1, y1)

    @classmethod
    def load(cls, directory: Path) -> "Profile":
        meta = json.loads((directory / "profile.json").read_text())
        block = meta["block"]
        thresholds = meta.get("thresholds", {})
        return cls(
            name=meta.get("name", directory.name),
            width=int(block["width"]),
            height=int(block["height"]),
            right=int(block["right"]),
            bottom=int(block["bottom"]),
            directory=directory,
            found=float(thresholds.get("present", 400)),
            clean=float(thresholds.get("clean", 250)),
            notes=meta.get("notes", ""),
            scaled=bool(block.get("scaled", False)),
            ref_width=int(block.get("refWidth", 1280)),
            right_ratio=float(block.get("rightRatio", 0.0)),
            bottom_ratio=float(block.get("bottomRatio", 0.0)),
        )

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "profile.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "block": {"width": self.width, "height": self.height, "right": self.right, "bottom": self.bottom, "scaled": self.scaled, "refWidth": self.ref_width, "rightRatio": self.right_ratio, "bottomRatio": self.bottom_ratio},
                    "thresholds": {"present": self.found, "clean": self.clean},
                    "notes": self.notes,
                },
                indent=2,
            )
            + "\n"
        )


def load_profiles(directory: Path = PROFILES_DIR) -> dict:
    profiles = {}
    if directory.is_dir():
        for meta in sorted(directory.glob("*/profile.json")):
            profile = Profile.load(meta.parent)
            profiles[profile.name] = profile
    return profiles


PROFILES = load_profiles()
AVITO = PROFILES["avito"]

# Names the fitting scripts and tests use.
LOGO_W, LOGO_H = AVITO.width, AVITO.height
LOGO_RIGHT, LOGO_BOTTOM = AVITO.right, AVITO.bottom
PRESENCE_FOUND, PRESENCE_CLEAN = AVITO.found, AVITO.clean


def profile_named(name: Optional[str]) -> Profile:
    key = (name or AVITO.name).strip().lower()
    if key not in PROFILES:
        raise ValueError(f"unknown profile '{name}': {', '.join(sorted(PROFILES))}, {AUTO}")
    return PROFILES[key]


def logo_box(width: int, height: int, profile: Profile = AVITO) -> tuple:
    return profile.box(width, height)


def template(profile: Profile = AVITO) -> Optional[Image.Image]:
    if not profile.template.exists():
        return None
    return Image.open(profile.template).convert("L")


def dilate(mask: Image.Image, radius: int = MASK_DILATE) -> Image.Image:
    return mask.filter(ImageFilter.MaxFilter(radius * 2 + 1)) if radius > 0 else mask


def logo_mask(width: int, height: int, kind: str = "logo", profile: Profile = AVITO) -> Image.Image:
    """White-on-black mask of the profile's watermark, dilated so a few pixels
    of misalignment still fall inside it."""
    x0, y0, x1, y1 = profile.box(width, height)
    mask = Image.new("L", (width, height), 0)
    silhouette = None if kind == "rect" else template(profile)
    if silhouette is None:
        mask.paste(255, (x0, y0, x1, y1))
    else:
        silhouette = silhouette.resize((x1 - x0, y1 - y0), Image.BILINEAR)
        mask.paste(silhouette.point(lambda v: 255 if v > 64 else 0), (x0, y0))
    return dilate(mask)


def box_mask(width: int, height: int, box: tuple, pad: int = 0) -> Image.Image:
    """Rectangular mask from (x0, y0, x1, y1), padded, clamped and dilated."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    x1, y1 = min(width, int(x1) + pad), min(height, int(y1) + pad)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("mask_box is empty or outside the image")
    mask = Image.new("L", (width, height), 0)
    mask.paste(255, (x0, y0, x1, y1))
    return dilate(mask)


def boxes_mask(width: int, height: int, boxes: list, pad: int = AUTO_PAD) -> Image.Image:
    """Union of padded rectangles."""
    mask = Image.new("L", (width, height), 0)
    for box in boxes:
        x0, y0, x1, y1 = box[:4]
        x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        x1, y1 = min(width, int(x1) + pad), min(height, int(y1) + pad)
        if x1 > x0 and y1 > y0:
            mask.paste(255, (x0, y0, x1, y1))
    return dilate(mask)


def parse_mask_box(text: str) -> tuple:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4 or not all(p.lstrip("-").isdigit() for p in parts):
        raise ValueError("mask_box must be x0,y0,x1,y1 in pixels")
    return tuple(int(p) for p in parts)


def work_window(width: int, height: int, profile: Profile = AVITO, mask: Optional[Image.Image] = None) -> tuple:
    """The part of the image LaMa actually processes: the mask's bounding box
    plus context."""
    bbox = mask.getbbox() if mask is not None else None
    x0, y0, x1, y1 = bbox if bbox else profile.box(width, height)
    return (max(0, x0 - WINDOW_PAD), max(0, y0 - WINDOW_PAD), min(width, x1 + WINDOW_PAD), min(height, y1 + WINDOW_PAD))


def normalized(values: list) -> list:
    """Zero-mean, unit-std copy, so contrast does not drive the score."""
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5 or 1.0
    return [(v - mean) / std for v in values]


class Detector:
    """Matched filter for a profile: `score = <highpass(corner), w>`. For the
    built-in Avito profile `w` is the mean difference between watermarked and
    cleaned corners; profiles made by fit_profile.py use diagonal LDA weights
    on contrast-normalised corners (`normalize` in detector.json), which
    tolerates a variable part such as a listing ID inside the mark."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.available = profile.detector.exists() and profile.detector_meta.exists()
        if not self.available:
            return
        meta = json.loads(profile.detector_meta.read_text())
        self.pad = int(meta.get("pad", DETECTOR_PAD))
        self.normalize = bool(meta.get("normalize", False))
        self.gain = float(meta.get("gain", 1.0))
        scale = float(meta["scale"])
        img = Image.open(profile.detector).convert("L")
        self.size = img.size
        self.w = [(v - 128) * scale / 127.0 for v in img.getdata()]

    def highpass(self, image: Image.Image) -> Optional[list]:
        p = self.profile
        x0, y0, x1, y1 = p.box(image.width, image.height)
        if (x1 - x0, y1 - y0) != p.block_size(image.width):
            return None
        k = p.scale(image.width)
        pad = max(1, round(self.pad * k))
        if x0 < pad or y0 < pad:
            return None
        gray = image.convert("L").crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        if gray.size != self.size:
            # Scaled profile on another width: bring the block to the
            # reference size so the filter lines up.
            gray = gray.resize(self.size, Image.BILINEAR)
        blurred = gray.filter(ImageFilter.GaussianBlur(6))
        return [a - b for a, b in zip(gray.getdata(), blurred.getdata())]

    def score(self, image: Image.Image) -> Optional[float]:
        if not self.available:
            return None
        hp = self.highpass(image)
        if hp is None:
            return None
        if self.normalize:
            hp = normalized(hp)
        return sum(a * b for a, b in zip(hp, self.w)) * self.gain


def _inv(v: int, k: int, b: int) -> int:
    if k >= 254:
        return v
    return int(min(255, max(0, round((v - b) * 255.0 / k))))


class Unblender:
    """Inverts a profile's fitted overlay: orig = (obs - b) / k per channel."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.available = profile.k_map.exists() and profile.b_map.exists()
        if self.available:
            self.k = Image.open(profile.k_map).convert("RGB")
            self.b = Image.open(profile.b_map).convert("RGB")
            if self.k.size != (profile.width, profile.height) or self.b.size != self.k.size:
                self.available = False

    def apply(self, image: Image.Image) -> tuple:
        """Returns the unblended image and a mask of pixels too opaque to
        invert (to hand to LaMa), or (image, full mask) when no maps."""
        p = self.profile
        x0, y0, x1, y1 = p.box(image.width, image.height)
        if not self.available or (x1 - x0, y1 - y0) != (p.width, p.height):
            return image, logo_mask(image.width, image.height, "logo", p)
        crop = image.crop((x0, y0, x1, y1))
        out = []
        opaque_px = []
        for (r, g, bl), (kr, kg, kb), (br, bg, bb) in zip(crop.getdata(), self.k.getdata(), self.b.getdata()):
            if min(kr, kg, kb) / 255.0 < 1.0 - LAMA_ALPHA_THRESHOLD:
                opaque_px.append(255)
                out.append((r, g, bl))
                continue
            opaque_px.append(0)
            out.append((_inv(r, kr, br), _inv(g, kg, bg), _inv(bl, kb, bb)))
        opaque = Image.new("L", (p.width, p.height))
        opaque.putdata(opaque_px)
        patch = Image.new("RGB", (p.width, p.height))
        patch.putdata(out)
        result = image.copy()
        result.paste(patch, (x0, y0))
        mask = Image.new("L", image.size, 0)
        mask.paste(opaque, (x0, y0))
        return result, dilate(mask, 1)


def ensure_yolo_weights(path: Path = YOLO_PATH) -> Path:
    """Downloads the generic detector once and checks its hash."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(YOLO_URL, tmp)
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != YOLO_SHA256:
        raise RuntimeError(f"unexpected YOLO weights hash {digest} at {path}")
    return path


class AutoDetector:
    """Generic watermark/logo boxes from the fine-tuned YOLO11x."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self.available = YOLO_PATH.exists()

    def _load(self):
        with self._lock:
            if self._model is None:
                from ultralytics import YOLO

                self._model = YOLO(str(ensure_yolo_weights()))
                self.available = True
        return self._model

    def boxes(self, image: Image.Image, conf: float = YOLO_CONF) -> list:
        """[(x0, y0, x1, y1, confidence)] in image pixels, highest first."""
        result = self._load().predict(image.convert("RGB"), verbose=False, conf=conf, imgsz=YOLO_IMGSZ)[0]
        found = []
        if result.boxes is not None:
            for box in result.boxes:
                x0, y0, x1, y1 = (int(round(v)) for v in box.xyxy[0].tolist())
                found.append((x0, y0, x1, y1, round(float(box.conf), 3)))
        return sorted(found, key=lambda b: -b[4])


class Cleaner:
    """Lazy LaMa wrapper; the model loads on the first request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lama = None
        self.device = "unloaded"
        self.detectors = {name: Detector(p) for name, p in PROFILES.items()}
        self.unblenders = {name: Unblender(p) for name, p in PROFILES.items()}
        self.auto = AutoDetector()
        self.requests = 0
        self.total_ms = 0.0

    @property
    def unblender(self) -> Unblender:
        return self.unblenders[AVITO.name]

    def _load(self):
        with self._lock:
            if self._lama is None:
                import torch
                from simple_lama_inpainting import SimpleLama
                from simple_lama_inpainting.models.model import LAMA_MODEL_URL
                from simple_lama_inpainting.utils import download_model

                # The TorchScript file was saved on CUDA; loading it anywhere
                # else needs map_location, which SimpleLama.__init__ omits.
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                lama = SimpleLama.__new__(SimpleLama)
                lama.model = torch.jit.load(download_model(LAMA_MODEL_URL), map_location=device)
                lama.model.eval()
                lama.model.to(device)
                lama.device = device
                self._lama = lama
                self.device = str(device)
        return self._lama

    def inpaint(self, source: Image.Image, mask: Image.Image, profile: Profile = AVITO) -> Image.Image:
        lama = self._load()
        window = work_window(source.width, source.height, profile, mask)
        window_mask = mask.crop(window)
        inpainted = lama(source.crop(window), window_mask)
        # LaMa pads to a multiple of 8; crop back to the window size, then
        # write back only the masked pixels.
        inpainted = inpainted.crop((0, 0, window[2] - window[0], window[3] - window[1]))
        result = source.copy()
        result.paste(inpainted, (window[0], window[1]), window_mask)
        return result

    def clean(self, image: Image.Image, kind: str = "logo", method: str = "auto", profile: Profile = AVITO) -> tuple:
        """Returns (cleaned image, method used) for a profile watermark."""
        started = time.time()
        source = image.convert("RGB")
        unblender = self.unblenders[profile.name]
        if method != "unblend" or kind == "rect" or not unblender.available:
            result = self.inpaint(source, logo_mask(source.width, source.height, kind, profile), profile)
            used = "lama"
        else:
            result, opaque = unblender.apply(source)
            if opaque.getbbox():
                result = self.inpaint(result, opaque, profile)
                used = "unblend+lama"
            else:
                used = "unblend"
        self.requests += 1
        self.total_ms += (time.time() - started) * 1000
        return result, used

    def clean_mask(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpaint a caller-supplied mask; no profile, no presence check."""
        started = time.time()
        source = image.convert("RGB")
        if mask.size != source.size:
            raise ValueError("mask size must match the image")
        if not mask.getbbox():
            raise ValueError("mask is empty")
        result = self.inpaint(source, mask)
        self.requests += 1
        self.total_ms += (time.time() - started) * 1000
        return result

    def clean_auto(self, image: Image.Image) -> tuple:
        """Detect marks anywhere with the generic detector and inpaint them.
        Returns (cleaned image or None when nothing was found, boxes)."""
        boxes = self.auto.boxes(image)
        if not boxes:
            return None, []
        mask = boxes_mask(image.width, image.height, boxes)
        return self.clean_mask(image, mask), boxes


cleaner = Cleaner()
detector = cleaner.detectors[AVITO.name]


def presence_score(image: Image.Image, profile: Profile = AVITO) -> Optional[float]:
    """Watermark presence score, or None when the detector cannot judge this image."""
    return cleaner.detectors[profile.name].score(image)


def profile_info(profile: Profile) -> dict:
    return {
        "name": profile.name,
        "block": {"width": profile.width, "height": profile.height, "right": profile.right, "bottom": profile.bottom, "scaled": profile.scaled, "refWidth": profile.ref_width, "rightRatio": profile.right_ratio, "bottomRatio": profile.bottom_ratio},
        "detector": cleaner.detectors[profile.name].available,
        "unblend": cleaner.unblenders[profile.name].available,
        "thresholds": {"present": profile.found, "clean": profile.clean},
        "notes": profile.notes,
    }


def build_app():
    from fastapi import FastAPI, Header, HTTPException, Query, Request, Response

    app = FastAPI(title="watermarkless", version="0.4.0")
    token = os.environ.get("CLEANER_TOKEN", "").strip()
    allow_anon = os.environ.get("CLEANER_ALLOW_ANON", "") == "1"
    if not token and not allow_anon:
        raise SystemExit("CLEANER_TOKEN is not set; set it or CLEANER_ALLOW_ANON=1 for a trusted network")

    def check_auth(authorization: Optional[str]) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="bad or missing bearer token")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": "lama",
            "device": cleaner.device,
            "profiles": sorted(PROFILES),
            "auto": cleaner.auto.available,
            "detector": detector.available,
            "unblend": cleaner.unblender.available,
            "requests": cleaner.requests,
            "avgMs": round(cleaner.total_ms / cleaner.requests) if cleaner.requests else None,
        }

    @app.get("/v1/profiles")
    def profiles():
        return {"profiles": [profile_info(p) for p in PROFILES.values()], "auto": cleaner.auto.available}

    @app.post("/v1/clean")
    async def clean(
        request: Request,
        profile: str = Query(default=AVITO.name),
        mask_box: Optional[str] = Query(default=None, pattern=r"^-?\d+,-?\d+,-?\d+,-?\d+$"),
        mask: str = Query(default="logo", pattern="^(logo|rect)$"),
        quality: int = Query(default=95, ge=1, le=100),
        method: str = Query(default="auto", pattern="^(auto|lama|unblend)$"),
        force: bool = Query(default=False),
        authorization: Optional[str] = Header(default=None),
    ):
        check_auth(authorization)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY:
            raise HTTPException(status_code=413, detail=f"body exceeds {MAX_BODY} bytes")
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            data = await upload.read() if hasattr(upload, "read") else b""
        else:
            data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="empty body")
        if len(data) > MAX_BODY:
            raise HTTPException(status_code=413, detail=f"body exceeds {MAX_BODY} bytes")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as error:  # noqa: BLE001 - any decode failure is a 400
            raise HTTPException(status_code=400, detail=f"not an image: {error}") from error
        if max(image.size) > MAX_SIDE:
            raise HTTPException(status_code=413, detail=f"image side exceeds {MAX_SIDE}px")
        image = image.convert("RGB")
        fmt = lambda v: "n/a" if v is None else f"{v:.0f}"  # noqa: E731

        if mask_box:
            try:
                custom = box_mask(image.width, image.height, parse_mask_box(mask_box))
                cleaned = cleaner.clean_mask(image, custom)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            headers = {"X-Mask-Box": ",".join(str(v) for v in custom.getbbox()), "X-Method": "lama"}
        elif profile.strip().lower() == AUTO:
            cleaned, boxes = cleaner.clean_auto(image)
            if cleaned is None:
                return Response(status_code=204, headers={"X-Method": "none"})
            headers = {"X-Mask-Box": ";".join(",".join(str(v) for v in b[:4]) for b in boxes), "X-Method": "auto+lama"}
        else:
            try:
                chosen = profile_named(profile)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            before = presence_score(image, chosen)
            box = ",".join(str(v) for v in chosen.box(image.width, image.height))
            if mask == "logo" and not force and before is not None and before < chosen.found:
                return Response(status_code=204, headers={"X-Mask-Box": box, "X-Watermark-Before": fmt(before), "X-Method": "none"})
            cleaned, used = cleaner.clean(image, mask, method, chosen)
            after = presence_score(cleaned, chosen)
            headers = {"X-Mask-Box": box, "X-Watermark-Before": fmt(before), "X-Watermark-After": fmt(after), "X-Method": used}
            if mask == "logo" and not force and after is not None and after > chosen.clean:
                raise HTTPException(status_code=422, detail=f"watermark still detectable after cleaning (score {after:.0f}, was {fmt(before)})", headers=headers)
        out = io.BytesIO()
        cleaned.save(out, format="JPEG", quality=quality, subsampling=0)
        return Response(content=out.getvalue(), media_type="image/jpeg", headers=headers)

    return app


def main(argv: list) -> int:
    if len(argv) >= 3 and argv[1] == "clean":
        # CLI check: app.py clean <input> [output] [profile|auto] [x0,y0,x1,y1]
        src = Path(argv[2])
        dst = Path(argv[3]) if len(argv) > 3 else src.with_name(src.stem + ".clean.jpg")
        which = argv[4] if len(argv) > 4 else AVITO.name
        image = Image.open(src).convert("RGB")
        if len(argv) > 5:
            cleaned = cleaner.clean_mask(image, box_mask(image.width, image.height, parse_mask_box(argv[5])))
            cleaned.save(dst, format="JPEG", quality=95, subsampling=0)
            print(f"{dst} method=lama mask_box={argv[5]} device={cleaner.device}")
            return 0
        if which == AUTO:
            cleaned, boxes = cleaner.clean_auto(image)
            if cleaned is None:
                print("nothing detected")
                return 1
            cleaned.save(dst, format="JPEG", quality=95, subsampling=0)
            print(f"{dst} method=auto+lama boxes={boxes} device={cleaner.device}")
            return 0
        chosen = profile_named(which)
        before = presence_score(image, chosen)
        cleaned, used = cleaner.clean(image, "logo", "auto", chosen)
        cleaned.save(dst, format="JPEG", quality=95, subsampling=0)
        print(f"{dst} profile={chosen.name} method={used} device={cleaner.device} before={before} after={presence_score(cleaned, chosen)}")
        return 0
    import uvicorn

    uvicorn.run(build_app(), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8765")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
