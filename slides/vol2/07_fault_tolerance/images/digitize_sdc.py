#!/usr/bin/env python3
"""Digitize the gradient-norm trace in dean-gradient-norm-spike.jpg.

The photo shows a slide from Jeff Dean's MLSys 2024 keynote (Google),
reproduced in Fault Tolerance, figure 9: a training job's gradient norm over
time with a boxed anomaly attributed to silent data corruption. The original
axes carry no tick values, so the trace is kept in relative units: time runs
0 to 1 across the plot, and the height above the photographed x axis is
scaled so that the median of the trace is 1. The output is the dot pattern as
photographed, sampled on a 3-pixel grid; nothing is interpolated or smoothed.

make_figs.py calls digitize() to draw sdc_trace.svg. Run directly, this
writes sdc_trace.csv (read by calcs.py) and prints the anomaly window:
python3 digitize_sdc.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

HERE = Path(__file__).parent
PHOTO = "dean-gradient-norm-spike.jpg"
Y0, Y1, X0, X1 = 250, 700, 140, 1280       # plot area in photo pixels
AXIS_Y = 715.0                              # photographed x axis line
BOX = ((756 - X0) / (X1 - X0), (885 - X0) / (X1 - X0))   # the slide's yellow box


def digitize(img):
    """Return (t, norm) for the dots in the keynote photo `img`."""
    lum = np.asarray(img.convert("L")).astype(float)
    bg = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(12))).astype(float)
    rgb = np.asarray(img.convert("RGB")).astype(int)
    # Gray dots are brighter than the local background; the slide draws the
    # later steps in blue, which a bluish tint separates from the background.
    dots = ((lum - bg) > 14) | ((rgb[..., 2] - rgb[..., 0] > 8) & (rgb[..., 2] > 125))
    mask = np.zeros_like(dots)
    mask[Y0:Y1, X0:X1] = dots[Y0:Y1, X0:X1]
    # Remove what the slide drew on top of the data: the callout label, the
    # yellow box, its connector curve, and the y axis line.
    for ya, yb, xa, xb in [(318, 378, 455, 715), (318, 334, 750, 895), (614, 632, 750, 895),
                           (318, 632, 750, 766), (318, 632, 876, 895), (365, 485, 575, 760),
                           (250, 560, 138, 168)]:
        mask[ya:yb, xa:xb] = False
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    mask &= np.isin(lab, 1 + np.nonzero(sizes >= 2)[0])   # drop single-pixel noise
    ys, xs = np.nonzero(mask)
    cells = np.unique(np.stack([ys // 3, xs // 3], 1), axis=0)   # one sample per 3 x 3 cell
    py, px = cells[:, 0] * 3 + 1, cells[:, 1] * 3 + 1
    t = (px - X0) / (X1 - X0)
    h = AXIS_Y - py
    order = np.argsort(t)
    return t[order], (h / np.median(h))[order]


if __name__ == "__main__":
    t, norm = digitize(Image.open(HERE / PHOTO))
    np.savetxt(HERE / "sdc_trace.csv", np.stack([t, norm], 1), delimiter=",",
               header="t_relative,gradient_norm_relative_to_median", fmt="%.4f", comments="")
    inside = (t >= BOX[0]) & (t <= BOX[1])
    print(f"samples: {len(t)}")
    print(f"anomaly box (relative time): {BOX[0]:.3f} to {BOX[1]:.3f}")
    print(f"peak inside box / median: {norm[inside].max():.1f}")
    print(f"peak outside box / median: {norm[~inside].max():.1f}")
