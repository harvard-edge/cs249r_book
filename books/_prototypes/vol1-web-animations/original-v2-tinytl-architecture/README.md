# Why does a frozen backbone still need a backward pass?

The original three-block comparison. Saved input tiles are schematic, not a memory-byte model. TinyTL freezes backbone weights and trains biases, lite residuals, and the classifier. Mask and normalization state may be retained or recomputed.

Frozen backbone weights never change, but backward traverses all three blocks. Only saved-input state is illustrated; masks, normalization, gradient storage, and total training memory are not quantified. Parameter updates occur only at optimizer step.

Run `python3 check.py` for the disposable actual-chapter fixture and browser checks. The static SVG, caption, and manuscript remain unchanged. Motion is illustrative; endpoint states determine the visible ownership and values.

Source context: `books/vol2/11_edge_intelligence/11_edge_intelligence.qmd`, anchor `fig-tinytl-architecture`. The original PNG/SVG and owning section were reviewed before implementation.

Validation: `check.py` renders the existing actual chapter and verifies all sequence endpoints at desktop 1440 and phone 390 in both chapter themes, controls, pause/resume, reduced motion, Expand, and print/no-JavaScript fallback. Fresh motion contexts capture paused viewport screenshots independently of full component captures. Review artifacts are in `/tmp/original-v2-tinytl-architecture-review/`. Production enrollment/build verification belongs to the integration wave.
