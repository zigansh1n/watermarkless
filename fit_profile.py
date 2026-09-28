"""Make a watermark profile from sample photos of one site.

    python fit_profile.py <name> <dir with photos> [more dirs...] [--box x0,y0,x1,y1]

What it does, in order:

1. Finds the mark. Runs the generic YOLO detector on up to 40 photos and
   takes the median box measured from the bottom-right corner (marks sit at
   a fixed offset there); `--box` gives that box by hand for one 1280-class
   photo instead. The block gets 6 px of margin.
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
BLOCK_MARGIN = 6
DETECT_SAMPLE = 40


def highpass_block(image, box, pad):
    x0, y0, x1, y1 = box
    if x0 < pad or y0 < pad or x1 > image.width or y1 > image.height:
        return None
    gray = image.convert("L").crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    return [a - b for a, b in zip(gray.getdata(), gray.filter(ImageFilter.GaussianBlur(6)).getdata())]


def summary(label, values):
    values = sorted(values)
    print(f"{label}: n={len(values)} min={values[0]:.0f} p5={values[len(values) // 20]:.0f} median={values[len(values) // 2]:.0f} max={values[-1]:.0f}")
    return values


def locate(photos, given):
    if given:
        x0, y0, x1, y1 = given
        image = Image.open(photos[0])
        return (x1 - x0, y1 - y0, image.width - x1, image.height - y1)
    widths, heights, rights, bottoms = [], [], [], []
    for path in photos[:DETECT_SAMPLE]:
        image = Image.open(path).convert("RGB")
        boxes = app.cleaner.auto.boxes(image)
        if not boxes:
            continue
        x0, y0, x1, y1, _ = boxes[0]
        widths.append(x1 - x0)
        heights.append(y1 - y0)
        rights.append(image.width - x1)
        bottoms.append(image.height - y1)
    if len(widths) < 5:
        raise SystemExit("the generic detector found the mark on fewer than 5 photos; pass --box")
    print(f"detector located the mark on {len(widths)} of {min(len(photos), DETECT_SAMPLE)} photos")
    return tuple(int(statistics.median(v)) for v in (widths, heights, rights, bottoms))


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

    width, height, right, bottom = locate(photos, given)
    width += 2 * BLOCK_MARGIN
    height += 2 * BLOCK_MARGIN
    right = max(0, right - BLOCK_MARGIN)
    bottom = max(0, bottom - BLOCK_MARGIN)
    profile = app.Profile(name=name, width=width, height=height, right=right, bottom=bottom, directory=app.PROFILES_DIR / name, found=400.0, clean=250.0, notes=f"fitted from {len(photos)} photos")
    print(f"block {width}x{height}, right {right}, bottom {bottom}")

    # Silhouette from the mean high-pass inside the block.
    acc = None
    used = 0
    for path in photos:
        image = Image.open(path).convert("RGB")
        hp = highpass_block(image, profile.box(image.width, image.height), 0)
        if hp is None:
            continue
        acc = hp if acc is None else [a + b for a, b in zip(acc, hp)]
        used += 1
    mean = [v / used for v in acc]
    peak = max(mean)
    sil = Image.new("L", (width, height))
    sil.putdata([255 if v > 0.4 * peak else 0 for v in mean])
    sil = sil.filter(ImageFilter.MaxFilter(5))
    profile.directory.mkdir(parents=True, exist_ok=True)
    sil.save(profile.template)
    print(f"silhouette from {used} photos: {sum(1 for v in sil.getdata() if v)} of {width * height} px")

    # Detector: positives vs LaMa-cleaned negatives, last directory held out.
    holdout = Path(args[-1]).resolve()
    pos, neg, tpos, tneg = [], [], [], []
    for path in photos:
        image = Image.open(path).convert("RGB")
        box = profile.box(image.width, image.height)
        hp = highpass_block(image, box, app.DETECTOR_PAD)
        if hp is None:
            continue
        cleaned = app.cleaner.inpaint(image, app.logo_mask(image.width, image.height, "logo", profile), profile)
        hn = highpass_block(cleaned, box, app.DETECTOR_PAD)
        test = holdout in path.resolve().parents and len(args) > 2
        (tpos if test else pos).append(hp)
        (tneg if test else neg).append(hn)
    n = len(pos[0])
    w = [sum(v[i] for v in pos) / len(pos) - sum(v[i] for v in neg) / len(neg) for i in range(n)]
    norm = sum(x * x for x in w) ** 0.5
    w = [x / norm for x in w]
    score = lambda hp: sum(a * b for a, b in zip(hp, w))  # noqa: E731
    p_scores = summary("logo", [score(h) for h in pos + tpos])
    n_scores = summary("clean", [score(h) for h in neg + tneg])
    gap_low, gap_high = n_scores[-1], p_scores[len(p_scores) // 20]
    if gap_high <= gap_low:
        print("WARNING: logo and clean scores overlap; the detector will be unreliable", file=sys.stderr)
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
