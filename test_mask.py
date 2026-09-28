"""Self-check for the mask geometry; runs without torch: python test_mask.py"""

from PIL import Image

from app import LOGO_BOTTOM, LOGO_H, LOGO_RIGHT, LOGO_W, PRESENCE_CLEAN, PRESENCE_FOUND, WINDOW_PAD, box_mask, logo_box, logo_mask, parse_mask_box, presence_score, profile_named, template, work_window


def test_box_is_fixed_size_in_the_corner():
    for w, h in [(1280, 854), (1280, 960), (720, 960), (640, 480)]:
        x0, y0, x1, y1 = logo_box(w, h)
        assert (x1, y1) == (w - LOGO_RIGHT, h - LOGO_BOTTOM)
        assert (x1 - x0, y1 - y0) == (LOGO_W, LOGO_H)
    assert logo_box(1280, 854) == (1171, 812, 1274, 849)


def test_tiny_images_clamp_inside():
    x0, y0, x1, y1 = logo_box(80, 30)
    assert 0 <= x0 < x1 <= 80 and 0 <= y0 < y1 <= 30


def test_mask_is_the_logo_silhouette_and_little_else():
    w, h = 1280, 960
    mask = logo_mask(w, h)
    x0, y0, x1, y1 = logo_box(w, h)
    inside = sum(1 for x in range(x0, x1) for y in range(y0, y1) if mask.getpixel((x, y)))
    # The silhouette (94x24 glyphs + 2 px) plus the 4 px dilation covers
    # roughly two thirds of the box, never the whole rectangle.
    assert 0.45 < inside / ((x1 - x0) * (y1 - y0)) < 0.9, inside
    total = sum(1 for v in mask.getdata() if v)
    assert total < (x1 - x0 + 20) * (y1 - y0 + 20)
    assert mask.getpixel((10, 10)) == 0
    # the glyph row itself is fully covered
    assert all(mask.getpixel((x, y0 + 15)) for x in range(x0 + 8, x1 - 12))


def test_window_contains_the_mask_and_stays_small():
    w, h = 1280, 960
    wx0, wy0, wx1, wy1 = work_window(w, h)
    x0, y0, x1, y1 = logo_box(w, h)
    assert wx0 <= x0 - 8 and wy0 <= y0 - 8 and wx1 == w and wy1 == h
    assert (wx1 - wx0) <= LOGO_W + 2 * WINDOW_PAD + LOGO_RIGHT
    assert work_window(300, 200) == (0, 0, 300, 200)


def test_presence_score_separates_logo_from_clean_corner():
    w, h = 640, 480
    plain = Image.new("RGB", (w, h), (120, 110, 100))
    assert presence_score(plain) is not None and abs(presence_score(plain)) < PRESENCE_CLEAN
    x0, y0, x1, y1 = logo_box(w, h)
    marked = plain.copy()
    overlay = Image.new("RGB", (LOGO_W, LOGO_H), (255, 255, 255))
    alpha = template().point(lambda v: 110 if v > 64 else 0)
    marked.paste(overlay, (x0, y0), alpha)
    # A flat synthetic overlay lacks the real logo's shadow, so it scores far
    # below a real photo (1300..1500), but still well above a clean corner.
    assert presence_score(marked) - presence_score(plain) > 100
    # the same overlay drawn 60 px away from the expected corner adds nothing
    elsewhere = plain.copy()
    elsewhere.paste(overlay, (x0 - 60, y0 - 60), alpha)
    assert abs(presence_score(elsewhere) - presence_score(plain)) < 20


def test_custom_box_mask_and_window():
    w, h = 800, 600
    mask = box_mask(w, h, parse_mask_box("100,50,220,90"))
    assert mask.getbbox() == (96, 46, 224, 94)  # dilated by 4 px
    assert mask.getpixel((150, 70)) == 255 and mask.getpixel((400, 300)) == 0
    assert work_window(w, h, mask=mask) == (0, 0, 416, 286)
    for bad in ("1,2,3", "a,b,c,d", "300,300,100,100"):
        try:
            box_mask(w, h, parse_mask_box(bad))
        except ValueError:
            continue
        raise AssertionError(bad)


def test_profiles():
    assert profile_named(None).name == "avito" and profile_named(" Avito ").name == "avito"
    try:
        profile_named("cian")
    except ValueError as error:
        assert "avito" in str(error)
    else:
        raise AssertionError("unknown profile accepted")


if __name__ == "__main__":
    test_box_is_fixed_size_in_the_corner()
    test_tiny_images_clamp_inside()
    test_mask_is_the_logo_silhouette_and_little_else()
    test_window_contains_the_mask_and_stays_small()
    test_presence_score_separates_logo_from_clean_corner()
    test_custom_box_mask_and_window()
    test_profiles()
    print("ok")
