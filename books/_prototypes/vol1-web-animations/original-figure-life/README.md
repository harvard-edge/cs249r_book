# Original convolution figure, animated in place

A local review prototype for Volume I, Figure 4 (`fig-cnn`). It uses the SVG
emitted from the existing TikZ drawing inside the rendered architecture chapter.
The manuscript, static drawing, caption, and the other animations are unchanged.

Diagram retains the original image. Explore retains its dashed input plane,
orange receptive field, blue connections, and violet output plane. Next separates
moving the window, combining weighted input contributions, and storing an output
activation. The original figure specifies no numerical inputs or weights, so
this preview does not invent numerical results. The dots show logical weighted
contributions, not hardware packets or measured transfer time.

Play runs once; Next performs one action and stops. Back, Restart, and exact
sequence jumps restore completed states. Keyboard hints remain visible. Reduced
motion skips interpolation, and print restores the original static image.
The prototype reads the shared build-time timing attributes already emitted by
the supplied chapter render; it has no independent playback-speed defaults.

Build the local review copy from a rendered chapter that includes the shared
playback metadata:

```sh
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
