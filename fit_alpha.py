"""Fit the logo overlay from a sample of watermarked photos.

    python fit_alpha.py <dir with photos> [more dirs...]

For every photo the background under the logo is approximated by LaMa
inpainting of the logo window; then, per pixel and per channel, a straight
line obs = k*bg + b is fitted across all photos. k is 1-alpha and b is
alpha*colour of the overlay. Writes k.png and b.png into the profile directory
(profiles/avito by default) and prints the fit quality. Photos whose corner does not look like it carries
the logo are skipped, so a directory may hold clean pictures too.
"""

import sys
from pathlib import Path

from PIL import Image

import app

MIN_SAMPLES = 40


def collect(paths):
    for root in paths:
        for path in sorted(Path(root).rglob("*.jpg")):
            if "clean" in path.parts:
                continue
            yield path


def main(argv):
    photos = list(collect(argv[1:]))
    if not photos:
        print("no photos given", file=sys.stderr)
        return 2
    box_w, box_h = app.LOGO_W, app.LOGO_H
    obs = []  # per photo: list of (r,g,b) over the logo box
    bgs = []
    skipped = 0
    for path in photos:
        image = Image.open(path).convert("RGB")
        x0, y0, x1, y1 = app.logo_box(image.width, image.height)
        score = app.presence_score(image)
        if (x1 - x0, y1 - y0) != (box_w, box_h) or (score is not None and score < app.PRESENCE_FOUND):
            skipped += 1
            continue
        background = app.cleaner.inpaint(image, app.logo_mask(image.width, image.height))
        obs.append(list(image.crop((x0, y0, x1, y1)).getdata()))
        bgs.append(list(background.crop((x0, y0, x1, y1)).getdata()))
        if len(obs) % 25 == 0:
            print(f"{len(obs)} samples...", flush=True)
    n = len(obs)
    print(f"samples: {n}, skipped: {skipped}")
    if n < MIN_SAMPLES:
        print(f"need at least {MIN_SAMPLES} samples", file=sys.stderr)
        return 1

    k_px = []
    b_px = []
    residuals = []
    for i in range(box_w * box_h):
        kk = []
        bb = []
        for ch in range(3):
            xs = [bgs[s][i][ch] for s in range(n)]
            ys = [obs[s][i][ch] for s in range(n)]
            mx = sum(xs) / n
            my = sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            k = sxy / sxx if sxx > 1e-6 else 1.0
            k = min(1.0, max(0.05, k))
            b = my - k * mx
            b = min(255.0, max(0.0, b))
            residuals.append(sum(abs(y - (k * x + b)) for x, y in zip(xs, ys)) / n)
            kk.append(int(round(k * 255)))
            bb.append(int(round(b)))
        k_px.append(tuple(kk))
        b_px.append(tuple(bb))
    k_img = Image.new("RGB", (box_w, box_h))
    k_img.putdata(k_px)
    b_img = Image.new("RGB", (box_w, box_h))
    b_img.putdata(b_px)
    k_img.save(app.AVITO.k_map)
    b_img.save(app.AVITO.b_map)
    alphas = [1 - min(px) / 255 for px in k_px]
    print(f"alpha max {max(alphas):.2f}, logo pixels (alpha>0.05): {sum(1 for a in alphas if a > 0.05)} of {len(alphas)}")
    print(f"mean fit residual: {sum(residuals) / len(residuals):.2f} levels")
    print(f"wrote {app.AVITO.k_map} and {app.AVITO.b_map}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
