#!/usr/bin/env python3
"""Cut multi-subject reference sheets into one square input image per subject.

Background is a flat light colour (estimated from the border); foreground = colour distance above a threshold,
dilated so thin attached parts (staffs, bows, lantern handles) stay with their owner, then connected components.

    tools/split_sheets.py test_inputs/sheets/*.png --out test_inputs
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image


def split(path, out_dir, thresh=22.0, join_px=12, min_area_frac=0.004, pad=0.12, size=1024):
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    H, W = im.shape[:2]
    border = np.concatenate([im[:4].reshape(-1, 3), im[-4:].reshape(-1, 3), im[:, :4].reshape(-1, 3),
                             im[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    fg = np.linalg.norm(im - bg, axis=-1) > thresh
    fg = ndi.binary_opening(fg, iterations=1)
    joined = ndi.binary_dilation(fg, iterations=join_px)
    lab, n = ndi.label(joined)
    objs = ndi.find_objects(lab)
    areas = ndi.sum(fg, lab, index=np.arange(1, n + 1))
    keep = [i for i in range(n) if areas[i] > min_area_frac * H * W]
    # reading order: rows (by centre y, banded), then x
    cents = {i: ((objs[i][0].start + objs[i][0].stop) / 2, (objs[i][1].start + objs[i][1].stop) / 2) for i in keep}
    band = H / 6
    keep.sort(key=lambda i: (round(cents[i][0] / band), cents[i][1]))
    outs = []
    stem = Path(path).stem.split("_", 1)[1] if "_" in Path(path).stem else Path(path).stem
    for k, i in enumerate(keep, 1):
        ys, xs = objs[i]
        # shrink back from the dilation to the real subject extent
        sub = (lab[ys, xs] == i + 1) & fg[ys, xs]
        yy, xx = np.nonzero(sub)
        y0, y1, x0, x1 = ys.start + yy.min(), ys.start + yy.max() + 1, xs.start + xx.min(), xs.start + xx.max() + 1
        side = int(max(y1 - y0, x1 - x0) * (1 + 2 * pad))
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        canvas = np.empty((side, side, 3), np.float32)
        canvas[:] = bg
        # paste only this subject's pixels (other subjects' pixels inside the box are replaced by background)
        mask_full = np.zeros((H, W), bool)
        mask_full[ys, xs] = lab[ys, xs] == i + 1
        sy0, sx0 = cy - side // 2, cx - side // 2
        ty0, ty1 = max(0, sy0), min(H, sy0 + side)
        tx0, tx1 = max(0, sx0), min(W, sx0 + side)
        region = im[ty0:ty1, tx0:tx1]
        m = mask_full[ty0:ty1, tx0:tx1, None]
        canvas[ty0 - sy0:ty1 - sy0, tx0 - sx0:tx1 - sx0] = np.where(m, region, bg)
        img = Image.fromarray(canvas.clip(0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
        name = f"{stem}_{k:02d}"
        img.save(Path(out_dir) / f"{name}.png")
        outs.append((name, int(y1 - y0), int(x1 - x0)))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+")
    ap.add_argument("--out", default="test_inputs")
    a = ap.parse_args()
    for s in a.sheets:
        outs = split(s, a.out)
        print(f"{Path(s).name}: {len(outs)} subjects " + ", ".join(f"{n} ({h}x{w}px)" for n, h, w in outs))


if __name__ == "__main__":
    main()
