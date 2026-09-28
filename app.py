"""watermarkless: removes watermarks from photos with LaMa inpainting.

A *profile* describes one known watermark: where its block sits, the mask
silhouette, a presence detector and, optionally, fitted overlay maps. The
built-in profile is the Avito listing logo. Without a profile a caller passes
its own mask (a box or a PNG) and the same inpainting runs on it, with no
presence check.

Default method is LaMa inpainting of the mask. `method=unblend` inverts the
profile's fitted overlay `obs = (1-a)*orig + a*c` (`fit_alpha.py`) and hands
only near-opaque pixels to LaMa; it is experimental, the maps fitted from
JPEGs leave speckle, so it is never chosen automatically.

Presence is decided per profile by a matched filter learned from samples
(`fit_detector.py`): high-passed corner dotted with the mean difference
between watermarked and cleaned corners. For Avito, on 219 photos the logo
scores 300..2600 and clean corners -320..210.

HTTP API
  GET  /health
  GET  /v1/profiles
  POST /v1/clean                -> image/jpeg
       body: raw image bytes (Content-Type image/*) or multipart field `file`
       query: profile=avito (default) | mask_box=x0,y0,x1,y1 (custom, no detector)
              mask=logo|rect (profile silhouette or its whole block)
              quality=1..100 (default 95), method=auto|lama|unblend (auto = lama)
       auth: Authorization: Bearer <CLEANER_TOKEN>; required unless CLEANER_ALLOW_ANON=1
       reply headers: X-Mask-Box, X-Watermark-Before, X-Watermark-After, X-Method
       422 when the watermark is still detectable after cleaning
       204 when no watermark is detected (image left unchanged)

Only a window around the mask is processed and only the masked pixels are
written back, so the rest of the photo changes only by the JPEG re-encode.
"""

import io
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent

MASK_DILATE = 4
# Context LaMa sees around the mask; enough for wood grain and tile lines.
WINDOW_PAD = 192
MAX_SIDE = 4096
MAX_BODY = 25 * 1024 * 1024
# Above this alpha the inverse amplifies JPEG noise more than it restores.
LAMA_ALPHA_THRESHOLD = 0.6
DETECTOR_PAD = 8


@dataclass(frozen=True)
class Profile:
    """One known watermark."""

    name: str
    # Block size and its offset from the right and bottom edges, in pixels;
    # the block is the same on every served variant of the profile's site.
    width: int
    height: int
    right: int
    bottom: int
    template: Path
    detector: Path
    detector_meta: Path
    k_map: Path
    b_map: Path
    # Matched-filter thresholds: present at or above `found`, clean at or
    # below `clean` after processing.
    found: float
    clean: float

    def box(self, image_w: int, image_h: int) -> tuple:
        """Bounding box (x0, y0, x1, y1) of the watermark block on an image."""
        x1 = image_w - self.right
        y1 = image_h - self.bottom
        return (max(0, x1 - self.width), max(0, y1 - self.height), x1, y1)


# Measured on 1280x854, 1280x960, 720x960 and 640x480: the Avito logo block is
# 103x37 px, 6 px from the right edge and 5 px from the bottom, on all of them.
# Inside it the glyphs are 94x24 at offset (3, 4): the official logo's alpha
# silhouette fitted to the mean high-pass of 208 photos; logo_mask_1280.png is
# that silhouette grown by 2 px for the shadow.
AVITO = Profile(
    name="avito",
    width=103,
    height=37,
    right=6,
    bottom=5,
    template=HERE / "logo_mask_1280.png",
    detector=HERE / "logo_detector.png",
    detector_meta=HERE / "logo_detector.json",
    k_map=HERE / "logo_k.png",
    b_map=HERE / "logo_b.png",
    found=400.0,
    clean=250.0,
)
PROFILES = {AVITO.name: AVITO}

# Backwards-compatible names used by the fitting scripts and tests.
LOGO_W, LOGO_H = AVITO.width, AVITO.height
LOGO_RIGHT, LOGO_BOTTOM = AVITO.right, AVITO.bottom
PRESENCE_FOUND, PRESENCE_CLEAN = AVITO.found, AVITO.clean
TEMPLATE_PATH, K_PATH, B_PATH = AVITO.template, AVITO.k_map, AVITO.b_map
DETECTOR_PATH, DETECTOR_META_PATH = AVITO.detector, AVITO.detector_meta


def profile_named(name: Optional[str]) -> Profile:
    key = (name or AVITO.name).strip().lower()
    if key not in PROFILES:
        raise ValueError(f"unknown profile '{name}': {', '.join(sorted(PROFILES))}")
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


def box_mask(width: int, height: int, box: tuple) -> Image.Image:
    """Custom rectangular mask from (x0, y0, x1, y1), clamped and dilated."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("mask_box is empty or outside the image")
    mask = Image.new("L", (width, height), 0)
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


class Detector:
    """Matched filter for a profile: `score = <highpass(corner), w>` with `w`
    the mean difference between watermarked and cleaned corners in the
    sample. Scene texture is uncorrelated with the mark, so a present mark
    scores hundreds and a clean corner stays around zero."""

    def __init__(self, profile: Profile = AVITO) -> None:
        self.profile = profile
        self.available = profile.detector.exists() and profile.detector_meta.exists()
        if not self.available:
            return
        meta = json.loads(profile.detector_meta.read_text())
        self.pad = int(meta.get("pad", DETECTOR_PAD))
        scale = float(meta["scale"])
        img = Image.open(profile.detector).convert("L")
        self.size = img.size
        self.w = [(v - 128) * scale / 127.0 for v in img.getdata()]

    def highpass(self, image: Image.Image) -> Optional[list]:
        x0, y0, x1, y1 = self.profile.box(image.width, image.height)
        pad = self.pad
        if (x1 - x0, y1 - y0) != (self.profile.width, self.profile.height) or x0 < pad or y0 < pad:
            return None
        gray = image.convert("L").crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        if gray.size != self.size:
            return None
        blurred = gray.filter(ImageFilter.GaussianBlur(6))
        return [a - b for a, b in zip(gray.getdata(), blurred.getdata())]

    def score(self, image: Image.Image) -> Optional[float]:
        if not self.available:
            return None
        hp = self.highpass(image)
        if hp is None:
            return None
        return sum(a * b for a, b in zip(hp, self.w))


def _inv(v: int, k: int, b: int) -> int:
    if k >= 254:
        return v
    return int(min(255, max(0, round((v - b) * 255.0 / k))))


class Unblender:
    """Inverts a profile's fitted overlay: orig = (obs - b) / k per channel."""

    def __init__(self, profile: Profile = AVITO) -> None:
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


class Cleaner:
    """Lazy LaMa wrapper; the model loads on the first request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lama = None
        self.device = "unloaded"
        self.detectors = {name: Detector(p) for name, p in PROFILES.items()}
        self.unblenders = {name: Unblender(p) for name, p in PROFILES.items()}
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


cleaner = Cleaner()
detector = cleaner.detectors[AVITO.name]


def presence_score(image: Image.Image, profile: Profile = AVITO) -> Optional[float]:
    """Watermark presence score, or None when the detector cannot judge this image."""
    return cleaner.detectors[profile.name].score(image)


def profile_info(profile: Profile) -> dict:
    return {
        "name": profile.name,
        "block": {"width": profile.width, "height": profile.height, "right": profile.right, "bottom": profile.bottom},
        "detector": cleaner.detectors[profile.name].available,
        "unblend": cleaner.unblenders[profile.name].available,
        "thresholds": {"present": profile.found, "clean": profile.clean},
    }


def build_app():
    from fastapi import FastAPI, Header, HTTPException, Query, Request, Response

    app = FastAPI(title="watermarkless", version="0.3.0")
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
            "template": AVITO.template.exists(),
            "unblend": cleaner.unblender.available,
            "detector": detector.available,
            "requests": cleaner.requests,
            "avgMs": round(cleaner.total_ms / cleaner.requests) if cleaner.requests else None,
        }

    @app.get("/v1/profiles")
    def profiles():
        return {"profiles": [profile_info(p) for p in PROFILES.values()]}

    @app.post("/v1/clean")
    async def clean(
        request: Request,
        profile: str = Query(default=AVITO.name),
        mask_box: Optional[str] = Query(default=None, pattern=r"^-?\d+,-?\d+,-?\d+,-?\d+$"),
        mask: str = Query(default="logo", pattern="^(logo|rect)$"),
        quality: int = Query(default=95, ge=1, le=100),
        method: str = Query(default="auto", pattern="^(auto|lama|unblend)$"),
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
        else:
            try:
                chosen = profile_named(profile)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            before = presence_score(image, chosen)
            box = ",".join(str(v) for v in chosen.box(image.width, image.height))
            if mask == "logo" and before is not None and before < chosen.found:
                return Response(status_code=204, headers={"X-Mask-Box": box, "X-Watermark-Before": fmt(before), "X-Method": "none"})
            cleaned, used = cleaner.clean(image, mask, method, chosen)
            after = presence_score(cleaned, chosen)
            headers = {"X-Mask-Box": box, "X-Watermark-Before": fmt(before), "X-Watermark-After": fmt(after), "X-Method": used}
            if mask == "logo" and after is not None and after > chosen.clean:
                raise HTTPException(status_code=422, detail=f"watermark still detectable after cleaning (score {after:.0f}, was {fmt(before)})", headers=headers)
        out = io.BytesIO()
        cleaned.save(out, format="JPEG", quality=quality, subsampling=0)
        return Response(content=out.getvalue(), media_type="image/jpeg", headers=headers)

    return app


def main(argv: list) -> int:
    if len(argv) >= 3 and argv[1] == "clean":
        # CLI check: app.py clean <input.jpg> [output.jpg] [auto|unblend|lama] [x0,y0,x1,y1]
        src = Path(argv[2])
        dst = Path(argv[3]) if len(argv) > 3 else src.with_name(src.stem + ".clean.jpg")
        method = argv[4] if len(argv) > 4 else "auto"
        image = Image.open(src).convert("RGB")
        if len(argv) > 5:
            cleaned = cleaner.clean_mask(image, box_mask(image.width, image.height, parse_mask_box(argv[5])))
            cleaned.save(dst, format="JPEG", quality=95, subsampling=0)
            print(f"{dst} method=lama mask_box={argv[5]} device={cleaner.device}")
            return 0
        before = presence_score(image)
        cleaned, used = cleaner.clean(image, "logo", method)
        cleaned.save(dst, format="JPEG", quality=95, subsampling=0)
        print(f"{dst} method={used} device={cleaner.device} before={before} after={presence_score(cleaned)}")
        return 0
    import uvicorn

    uvicorn.run(build_app(), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8765")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
