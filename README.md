# watermarkless

Removes watermarks from photos with LaMa inpainting. Three ways to say where
the mark is: a fitted *profile* for a known site (Avito is built in), `auto`
with a generic YOLO watermark detector for any site, or an explicit box.
Two front ends over the same code: an HTTP service (`app.py`, what Avito
Manager calls from `watermark_cleaner.rs` when a download is asked to strip
the watermark) and an MCP server (`mcp_server.py`) for agents working with
local files.

## Run

Docker (CPU, 2.3 GB image, weights baked in and pinned by SHA-256; built and
smoke-tested 2026-09-28):

```bash
docker build -t watermarkless .
docker run -d --name watermark-cleaner -p 8765:8765 -e CLEANER_TOKEN=change-me watermarkless
```

Bare Python (CUDA when available, otherwise CPU; the TorchScript model does not load on Apple MPS):

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
CLEANER_TOKEN=change-me .venv/bin/python app.py
```

The container speaks plain HTTP. Avito Manager accepts `http://` only for
localhost, so a remote host must sit behind a TLS reverse proxy (Caddy,
nginx, Traefik) and be entered as `https://cleaner.example`. Then in Avito
Manager → Settings → Automation → Watermark cleaner: URL and token.

If the service is unreachable, a download tries it once (with one retry),
records the failure on that photo and skips cleaning for the rest of the
listing, so an outage costs seconds, not a two-minute timeout per photo.

## MCP server

Requires Python 3.10+ (the `mcp` SDK). Stdio transport; register in Claude
Code with

```bash
claude mcp add -s user watermarkless -- /path/to/repo/.venv/bin/python /path/to/repo/mcp_server.py
```

Tools:

- `cleaner_health()` — model, detector and device state.
- `list_profiles()` — known watermarks with block geometry and thresholds, and whether `auto` is available.
- `find_watermarks(path, confidence=0.25)` — generic detector boxes for any site.
- `detect_watermark(path, profile="avito")` — presence score of one photo
  (avito: present ≥ 260, clean ≤ 250).
- `clean_photo(path, output_path?, method="lama"|"unblend", quality=95,
  profile="avito", mask_box?)` — cleans one file into `<dir>/clean/<name>.jpg`
  unless `output_path` is given; answers `no_watermark` / `still_present`
  instead of writing when the detector says so; `force=true` cleans anyway.
  `profile="auto"` finds marks of any site with the generic detector. With `mask_box=[x0,y0,x1,y1]` (image
  pixels) exactly that area is inpainted, no check.
- `clean_folder(directory, method, quality, skip_existing=true, profile, mask_box?)`
  — every image in a folder, not recursive, `clean/` ignored; per-file statuses.

Originals are never modified. The model loads on the first call (about a
second), then ~0.3 s per photo on an Apple M3 Max CPU.

## API

- `GET /health` → `{"status":"ok","model":"lama","device":"cpu|cuda|unloaded","profiles":[...],"detector":true,"requests":N,"avgMs":M}`
- `GET /v1/profiles` → known watermark profiles.
- `POST /v1/clean` with the image as the raw body (or multipart `file`),
  optional `?profile=avito|auto&mask=logo|rect&quality=95&method=auto|lama|unblend`,
  or `?mask_box=x0,y0,x1,y1` to inpaint a custom rectangle (any site, no
  presence check),
  `Authorization: Bearer <token>`. The token is required; set
  `CLEANER_ALLOW_ANON=1` only on a trusted network. Returns `image/jpeg` with
  headers `X-Mask-Box`, `X-Watermark-Before`, `X-Watermark-After`, `X-Method`.
  `204` when no logo is detected in the corner (the photo is left alone, the
  client keeps the original), `422` when the logo is still detectable after
  cleaning, `400` for non-images, `401` on a bad token, `413` above 25 MB or
  4096 px.

## Profiles, auto mode and custom masks

A **profile** is one known watermark near the bottom-right corner:
`profiles/<name>/profile.json` (block geometry and thresholds), `mask.png`
(silhouette or the whole block), optional `detector.png` + `detector.json`
(presence filter) and `k.png`/`b.png` (overlay maps). The block geometry
follows the photo size in one of three ways: fixed pixels (`avito`: 103x37
at 6/5 px on every variant), everything scaled with the width (`scaled`), or
a fixed-size block with margins proportional to width and height
(`rightRatio`/`bottomRatio`; `cian`: ~205x90 px, 10.3% of the width from
the right, 8.3% of the height from the bottom). Built in: `avito`, `cian`.
Make a new one from 40–200 sample photos of the site with one command:

```bash
.venv/bin/python fit_profile.py cian ~/photos/cian-listing-1 ~/photos/cian-listing-2
```

It locates the mark with the generic detector (or takes `--box x0,y0,x1,y1`),
picks the geometry model whose margins vary least across photo sizes,
averages the high-passed block to get the silhouette (or masks the whole
block when the mark has a variable part, such as Cian's listing ID),
inpaints every sample to make negatives and fits a presence filter with
thresholds from the score gap. When watermarked and clean scores overlap
(Cian: the ID digits differ on every listing) no detector is written and the
profile cleans every photo by its mask without the 204/422 gate. Restart the
service to load the profile.

**`auto`** uses a fine-tuned YOLO11x watermark detector
([corzent/yolo11x_watermark_detection](https://huggingface.co/corzent/yolo11x_watermark_detection),
MIT, 114 MB, downloaded on first use into `weights/` and checked by SHA-256).
Every box is padded by 10 px and inpainted. Measured on our photos: 66 of 70
Avito logos found with correct boxes, 2 false boxes on 40 clean photos,
~0.4 s per photo on CPU, and it found the Cian mark on a sample unseen by
anyone. Use it for sites without a profile; a profile is exact and 10x faster.

**`mask_box`** inpaints exactly the rectangle you pass, no detection.

## Method

1. **LaMa inpainting** of the logo silhouette (default, `method=lama`). Only
   a window of the logo plus 192 px of context is processed and only the
   masked pixels are written back.
2. **Reverse alpha blending** (`method=unblend`, experimental). The logo is
   `obs = (1-a)·orig + a·c` with the same `a` and `c` on every photo;
   `fit_alpha.py <photo dirs>` fits both per pixel and channel from a sample
   and writes `logo_k.png` / `logo_b.png`; cleaning inverts the formula and
   hands near-opaque pixels to LaMa. It restores real structure under the
   logo (board lines continue), but the maps fitted from 176 JPEG photos are
   too noisy: the result carries coloured speckle, so it is never chosen
   automatically. It needs a cleaner fit (shared alpha across channels,
   smoothing, chroma-aware) before it can be the default.

## Verification

`logo_detector.png` is a matched filter learned by `fit_detector.py`: the
mean high-passed corner of watermarked photos minus that of their LaMa-cleaned
versions. `presence_score` is the dot product of a photo's high-passed corner
with it. Measured on 202 photos from 9 listings plus 17 held-out photos:
logo 300..2600 (median ~1500), cleaned corners and logo-free scene crops
-320..210. Below the profile's `present` threshold (avito: 260; a field
run on 3131 photos found logos on white walls scoring 292..335, so the
first value of 400 was too high) the service answers 204 and the client keeps
the original; above `clean` (250) after cleaning it answers 422. `force=1`
(HTTP) or `force=true` (MCP) cleans the profile mask regardless. So if Avito moves or resizes the logo, downloads keep working
and report "not found" instead of damaging corners. A plain brightness test
was tried first and rejected: it missed 26 of 202 logos on bright walls. The
community YOLOv8n-seg from platonator777/Avito_watermarks (trained on
synthetic overlays, mAP50 0.98 on its own validation) found only 104 of 208
real logos at confidence 0.5 (118 at 0.25) with no false positives, at 40 ms
per image; the matched filter finds all 208, so it stays.

## Mask

`logo_mask_1280.png` is the alpha silhouette of the official Avito logo
(the `avito_logo_original.png` asset published in
platonator777/Avito_watermarks), scaled to 94x24 px and placed at (3, 4)
inside the logo block by maximising its overlap with the mean high-passed
corner of 208 real photos, then grown by 2 px for the drop shadow. The block
itself is a fixed 103x37 px, 6 px from the right edge and 5 px from the
bottom, on every served variant (measured on 1280x854, 1280x960, 720x960 and
640x480), so nothing is scaled; at clean time the mask is dilated by 4 px
more. Compared with the earlier blob derived from an antiznak diff it
inpaints a third fewer pixels (2069 vs 3083) and keeps edges that cross the
logo (a wall corner, a chair leg) instead of smearing them. Only a window of the logo plus 192 px of
context goes through LaMa and only the masked pixels are written back. If Avito
moves or resizes the logo, re-measure and update the constants at the top of
`app.py`; `python test_mask.py` checks the geometry without loading the model.

Use the server or the MCP tools for batches: the model loads once. Running
`app.py clean` per photo reloads it every time and costs about a second
extra per photo. Cost per photo on an Apple M3 Max, CPU only: about 0.3 s and a 1.6 GB resident
process (model 196 MB on disk, weights loaded once). In the container the
image defaults to `OMP_NUM_THREADS=4`: with one torch thread per core the
small inpainting window thrashed (11 s per photo on a 14-core VM, 1–3 s with
4 threads, 650 MiB resident). A full-frame pass, which
the first version did, took 5–12 s and peaked at 13.5 GB on 1280x960.

## Limits

Inpainting invents pixels: whatever was under the logo (small text, a face) is
not restored, only made plausible. Photos narrower than ~200 px get a mask a
few pixels wide and are effectively left alone.
