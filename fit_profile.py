"""Make a watermark profile from sample photos of one site.

    python fit_profile.py <name> <dir with photos> [more dirs...] [--box x0,y0,x1,y1]

What it does, in order:

1. Finds the mark. Runs the generic YOLO detector on up to 40 photos and
   takes the median box measured from the bottom-right corner, normalised
   to a 1280 px wide photo. If the offsets in pixels vary less than the
   offsets as a fraction of the width, the profile is fixed (Avito); else it
   scales with the width (Cian). `--box` gives the box by hand for one
   1280-wide photo instead. The block gets 6 px of margin.
2. Builds the silhouette: the mean high-passed block over all photos (the
   mark is the same on every photo, the scene is not) thresholded at 40 % of
   its peak and grown by 2 px. Written to profiles/<name>/mask.png.
3. Fits the presence detector exactly like the built-in one: each photo's
   block is inpainted with LaMa to make its negative; the detector is the
   mean high-passed block of positives minus negatives. Thresholds come from
   the score distributions (midpoints with margin).
4. Writes profiles/<name>/profile.json. The service picks it up on restart.

Needs 40+ photos; 100–200 are better. Photos where the detector finds no
box are used for the silhouette only when --box is given.
"""

import json
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageFilter

import app

MIN_PHOTOS = 40
BLOCK_MARGIN = 10
# Below this share of the block the silhouette is too sparse to trust (a
# mark with a variable part such as a listing ID): mask the whole block.
MIN_SILHOUETTE = 0.5
DETECT_SAMPLE = 40


def highpass_block(image, box, pad, size=None):
    """High-passed block crop with `pad`, resized to `size` when given."""
    x0, y0, x1, y1 = box
    if x0 < pad or y0 < pad or x1 > image.width or y1 > image.height:
        return None
    gray = image.convert("L").crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    if size and gray.size != size:
        gray = gray.resize(size, Image.BILINEAR)
    return [a - b for a, b in zip(gray.getdata(), gray.filter(ImageFilter.GaussianBlur(6)).getdata())]


def summary(label, values):
    values = sorted(values)
    print(f"{label}: n={len(values)} min={values[0]:.0f} p5={values[len(values) // 20]:.0f} median={values[len(values) // 2]:.0f} max={values[-1]:.0f}")
    return values


REF_WIDTH = 1280
REF_HEIGHT = 960


def spread(values):
    values = sorted(values)
    return values[3 * len(values) // 4] - values[len(values) // 4]


def locate(photos, given):
    """Block geometry: (width, height, right, bottom, scaled, right_ratio, bottom_ratio)."""
    if given:
        x0, y0, x1, y1 = given
        image = Image.open(photos[0])
        return (x1 - x0, y1 - y0, image.width - x1, image.height - y1, False, 0.0, 0.0)
    rows = []
    for path in photos[:DETECT_SAMPLE]:
        image = Image.open(path).convert("RGB")
        W, H = image.size
        # Union of every box in the bottom-right quarter: the detector often
        # splits a mark into its logo and its text line.
        boxes = [b for b in app.cleaner.auto.boxes(image, conf=0.2) if b[0] > W * 0.5 and b[1] > H * 0.5]
        if not boxes:
            continue
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        rows.append((x1 - x0, y1 - y0, W - x1, H - y1, W, H))
    if len(rows) < 5:
        raise SystemExit("the generic detector found the mark on fewer than 5 photos; pass --box")
    print(f"detector located the mark on {len(rows)} of {min(len(photos), DETECT_SAMPLE)} photos")
    # Three models of how the margins follow the photo size, scored by the
    # spread of the margins expressed in pixels of a 1280x960 photo.
    fixed = spread([r[2] for r in rows]) + spread([r[3] for r in rows])
    scaled = spread([r[2] * REF_WIDTH / r[4] for r in rows]) + spread([r[3] * REF_WIDTH / r[4] for r in rows])
    ratio = spread([r[2] / r[4] * REF_WIDTH for r in rows]) + spread([r[3] / r[5] * REF_HEIGHT for r in rows])
    print(f"margin spread: fixed {fixed:.0f}, width-scaled {scaled:.0f}, ratio {ratio:.0f} px")
    med = lambda values: statistics.median(values)  # noqa: E731
    width, height = int(med([r[0] for r in rows])), int(med([r[1] for r in rows]))
    if ratio <= min(fixed, scaled):
        print("-> block of fixed size, margins proportional to width/height")
        return (width, height, 0, 0, False, med([r[2] / r[4] for r in rows]), med([r[3] / r[5] for r in rows]))
    if scaled < fixed:
        print("-> everything scales with the width")
        k = [REF_WIDTH / r[4] for r in rows]
        return (int(med([r[0] * k[i] for i, r in enumerate(rows)])), int(med([r[1] * k[i] for i, r in enumerate(rows)])), int(med([r[2] * k[i] for i, r in enumerate(rows)])), int(med([r[3] * k[i] for i, r in enumerate(rows)])), True, 0.0, 0.0)
    print("-> fixed pixels")
    return (width, height, int(med([r[2] for r in rows])), int(med([r[3] for r in rows])), False, 0.0, 0.0)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    given = None
    if "--box" in argv:
        given = app.parse_mask_box(argv[argv.index("--box") + 1])
        args = [a for a in args if a != argv[argv.index("--box") + 1]]
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    name = args[0].strip().lower()
    photos = [p for root in args[1:] for p in sorted(Path(root).rglob("*.jpg")) if "clean" not in p.parts]
    if len(photos) < MIN_PHOTOS:
        print(f"need at least {MIN_PHOTOS} photos, got {len(photos)}", file=sys.stderr)
        return 1

    width, height, right, bottom, use_scaled, right_ratio, bottom_ratio = locate(photos, given)
    # Grow the block by the margin on every side, but never past the edge.
    width += BLOCK_MARGIN + min(BLOCK_MARGIN, max(0, right))
    height += BLOCK_MARGIN + min(BLOCK_MARGIN, max(0, bottom))
    right = max(0, right - BLOCK_MARGIN)
    bottom = max(0, bottom - BLOCK_MARGIN)
    profile = app.Profile(name=name, width=width, height=height, right=right, bottom=bottom, directory=app.PROFILES_DIR / name, found=400.0, clean=250.0, notes=f"fitted from {len(photos)} photos", scaled=use_scaled, ref_width=REF_WIDTH, right_ratio=round(right_ratio, 4), bottom_ratio=round(bottom_ratio, 4))
    print(f"block {width}x{height}, right {right} + {right_ratio:.3f}W, bottom {bottom} + {bottom_ratio:.3f}H{' (scaled)' if use_scaled else ''}")

    # Silhouette from the mean high-pass inside the block.
    acc = None
    used = 0
    for path in photos:
        image = Image.open(path).convert("RGB")
        hp = highpass_block(image, profile.box(image.width, image.height), 0, (width, height))
        if hp is None:
            continue
        acc = hp if acc is None else [a + b for a, b in zip(acc, hp)]
        used += 1
    mean = [v / used for v in acc]
    peak = max(mean)
    sil = Image.new("L", (width, height))
    sil.putdata([255 if v > 0.4 * peak else 0 for v in mean])
    sil = sil.filter(ImageFilter.MaxFilter(5))
    covered = sum(1 for v in sil.getdata() if v) / (width * height)
    profile.directory.mkdir(parents=True, exist_ok=True)
    if covered < MIN_SILHOUETTE:
        print(f"silhouette covers only {covered:.0%} of the block (variable mark content): masking the whole block")
        sil = Image.new("L", (width, height), 255)
    else:
        print(f"silhouette from {used} photos: {covered:.0%} of the block")
    sil.save(profile.template)

    # Detector: positives vs LaMa-cleaned negatives, last directory held out.
    holdout = Path(args[-1]).resolve()
    pos, neg, tpos, tneg = [], [], [], []
    for path in photos:
        image = Image.open(path).convert("RGB")
        box = profile.box(image.width, image.height)
        k = profile.scale(image.width)
        pad = max(1, round(app.DETECTOR_PAD * k))
        det_size = (width + 2 * app.DETECTOR_PAD, height + 2 * app.DETECTOR_PAD)
        hp = highpass_block(image, box, pad, det_size)
        if hp is None:
            continue
        cleaned = app.cleaner.inpaint(image, app.logo_mask(image.width, image.height, "logo", profile), profile)
        hn = highpass_block(cleaned, box, pad, det_size)
        test = holdout in path.resolve().parents and len(args) > 2
        (tpos if test else pos).append(hp)
        (tneg if test else neg).append(hn)
    n = len(pos[0])
    # Matched filter: mean high-passed block of positives minus negatives.
    # A diagonal-LDA variant with per-photo normalisation was tried and
    # separated worse on the Avito sample (logo min -1899 vs clean max 0,
    # against 300 vs 203 for this one), so the plain difference stays.
    w = [sum(v[i] for v in pos) / len(pos) - sum(v[i] for v in neg) / len(neg) for i in range(n)]
    norm = sum(x * x for x in w) ** 0.5
    w = [x / norm for x in w]
    score = lambda hp: sum(a * b for a, b in zip(hp, w))  # noqa: E731
    p_scores = summary("logo", [score(h) for h in pos + tpos])
    n_scores = summary("clean", [score(h) for h in neg + tneg])
    gap_low, gap_high = n_scores[-1], p_scores[len(p_scores) // 20]
    if gap_high <= gap_low:
        # No usable gap: a detector here would refuse real marks and pass
        # leftovers. The profile still cleans by its mask, without gating.
        print("logo and clean scores overlap: no presence detector written, the profile cleans every photo by its mask", file=sys.stderr)
        for stale in (profile.detector, profile.detector_meta):
            if stale.exists():
                stale.unlink()
        profile = app.Profile(**{**profile.__dict__, "notes": profile.notes + "; no presence detector (scores overlapped)"})
        profile.save()
        print(f"wrote {profile.directory}/ (profile.json, mask.png); restart the service to load it")
        return 0
    found = round(gap_low + 0.6 * (gap_high - gap_low))
    clean = round(gap_low + 0.3 * (gap_high - gap_low))
    scale = max(abs(x) for x in w)
    img = Image.new("L", (width + 2 * app.DETECTOR_PAD, height + 2 * app.DETECTOR_PAD))
    img.putdata([int(round(128 + 127 * x / scale)) for x in w])
    img.save(profile.detector)
    profile.detector_meta.write_text(json.dumps({"pad": app.DETECTOR_PAD, "scale": scale}))
    profile = app.Profile(**{**profile.__dict__, "found": float(found), "clean": float(clean)})
    profile.save()
    print(f"thresholds: present >= {found}, clean <= {clean}")
    print(f"wrote {profile.directory}/ (profile.json, mask.png, detector.png, detector.json); restart the service to load it")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
