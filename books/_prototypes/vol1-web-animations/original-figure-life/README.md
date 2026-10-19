# Original convolution figure, animated in place

A local review prototype for Volume I, Figure 4 (`fig-cnn`). It uses the SVG
emitted from the existing TikZ drawing inside the rendered architecture chapter.
The manuscript, static drawing, caption, and the other animations are unchanged.

Diagram retains the original image. Explore retains its dashed input plane,
orange receptive field, blue connections, and violet output plane. Next separates
moving the window, combining weighted input contributions, and storing an output
activation. The original static figure specifies no numerical inputs or weights. Explore
adds the nearby chapter's vertical-line example directly to the drawing: every
input cell carries its value, a shared-filter block sits in the diagram, and
completed activations are written into the violet output cells. The output
plane is enlarged uniformly, preserving parallel axes in both planes. A sum appears in the drawing
after Combine completes; the output value appears only after Write completes.
A visible calculation strip links one chosen input row to the matching filter
row. It shows the products for that row and, after Combine, the addition of
all row totals. The short calculation sits beside the drawing on desktop and
below it on phones. Full arithmetic remains available in a collapsed Inspect the
calculation section below the diagram.
The dots show logical weighted
contributions, not hardware packets or measured transfer time.

Play runs once; Next performs one action and stops. Back, Restart, and exact
sequence jumps restore completed states. Keyboard hints remain visible. Reduced
motion skips interpolation, and print restores the original static image.
The prototype reads the shared build-time timing attributes already emitted by
the supplied chapter render; it has no independent playback-speed defaults.

Build the local review copy from a rendered chapter that includes the shared
playback metadata:

```sh
python3 books/_prototypes/vol1-web-animations/original-figure-life/example-values.py
python3 books/_prototypes/vol1-web-animations/original-figure-life/build-preview.py /path/to/output/vol1/06_nn_architectures/06_nn_architectures.html
python3 -m http.server 8882 --bind 127.0.0.1 --directory books
```

Open `http://127.0.0.1:8882/_build/original-figure-life/vol1/06_nn_architectures/06_nn_architectures.html#fig-cnn`.
The builder creates a disposable copy under `books/_build/original-figure-life`;
it does not patch the supplied build or any published page.

Run the browser checks with `python3 books/_prototypes/vol1-web-animations/original-figure-life/check-preview.py`.
These checks exercise desktop and phone layouts in both themes, pause/resume,
manual stepping, sequence jumps, output counts, connection/window alignment,
reduced motion, and print fallback. This is an author-review prototype, not a
volume-wide implementation or a student study.


`example-values.py` owns the illustrative inputs and computes every patch,
product, row sum, and output in `example-values.json`. Regenerate the JSON before
building after an example change. No hardware specifications are introduced.
The additional browser checks in `check-values.py` verify every input and filter
label, every output position, sum/output commit timing, calculation disclosure
without state changes, and preservation of the original SVG paths at desktop
and phone widths in both themes.


`check-geometry.py` checks that both grid-axis directions remain parallel and
that blue connectors meet the moving window throughout interpolation. It also
checks pause/resume and each row selector without advancing the lesson, and
exports intermediate motion frames plus chapter and calculation screenshots.
Window motion and connector endpoints share one interpolation clock; SVG path
interpolation support is not required.
