"""Fit the logo presence detector from watermarked photos.

    python fit_detector.py <dir with photos> [more dirs...]

Each photo's corner is cleaned with LaMa to make its negative; the detector is
the mean high-passed corner of the positives minus that of the negatives,
saved as logo_detector.png (+ logo_detector.json with the scale). The last
listing directory is held out and its separation is printed.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageFilter

import app

PAD = app.DETECTOR_PAD


def highpass(image):
    x0, y0, x1, y1 = app.logo_box(image.width, image.height)
    if (x1 - x0, y1 - y0) != (app.LOGO_W, app.LOGO_H) or x0 < PAD or y0 < PAD:
        return None
    gray = image.convert("L").crop((x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD))
    return [a - b for a, b in zip(gray.getdata(), gray.filter(ImageFilter.GaussianBlur(6)).getdata())]


def summary(label, values):
    values = sorted(values)
    print(f"{label}: n={len(values)} min={values[0]:.0f} median={values[len(values) // 2]:.0f} max={values[-1]:.0f}")


def main(argv):
    dirs = sorted({p.parent for root in argv[1:] for p in Path(root).rglob("*.jpg") if "clean" not in p.parts})
    if len(dirs) < 2:
        print("need photos from at least two listing directories", file=sys.stderr)
        return 2
    holdout = dirs[-1]
    pos, neg, test_pos, test_neg = [], [], [], []
    for directory in dirs:
        for path in sorted(directory.glob("*.jpg")):
            image = Image.open(path).convert("RGB")
            hp = highpass(image)
            if hp is None:
                continue
            cleaned, _ = app.cleaner.clean(image, "logo", "lama")
            hn = highpass(cleaned)
            (test_pos if directory == holdout else pos).append(hp)
            (test_neg if directory == holdout else neg).append(hn)
        print(f"{directory.name}: done", flush=True)
    n = len(pos[0])
    w = [sum(v[i] for v in pos) / len(pos) - sum(v[i] for v in neg) / len(neg) for i in range(n)]
    norm = sum(x * x for x in w) ** 0.5
    w = [x / norm for x in w]
    score = lambda hp: sum(a * b for a, b in zip(hp, w))  # noqa: E731
    summary("train logo", [score(h) for h in pos])
    summary("train clean", [score(h) for h in neg])
    summary("held-out logo", [score(h) for h in test_pos])
    summary("held-out clean", [score(h) for h in test_neg])
    scale = max(abs(x) for x in w)
    img = Image.new("L", (app.LOGO_W + 2 * PAD, app.LOGO_H + 2 * PAD))
    img.putdata([int(round(128 + 127 * x / scale)) for x in w])
    img.save(app.DETECTOR_PATH)
    app.DETECTOR_META_PATH.write_text(json.dumps({"pad": PAD, "scale": scale}))
    print(f"wrote {app.DETECTOR_PATH.name}; thresholds in app.py: found >= {app.PRESENCE_FOUND:.0f}, clean <= {app.PRESENCE_CLEAN:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
