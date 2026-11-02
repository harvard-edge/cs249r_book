# When is a gradient ready for communication and for the optimizer?

The original four-layer backward order L4 → L1. One gradient transfer per illustrated layer, in the same collective order on all ranks. Timeline widths and motion are schematic, not a performance model; o occupies the GPU before launch.

AllReduce launches only after gradient readiness and GPU overhead, in fixed L4-to-L1 order. Optimizer waits for all reductions. Hiding a transfer depends on later compute and completion timing; no claim that every real workload hides all but the last bucket.

Run `python3 check.py` for the disposable actual-chapter fixture and browser checks. The static SVG, caption, and manuscript remain unchanged. Motion is illustrative; endpoint states determine the visible ownership and values.

Source context: `books/vol2/05_collective_communication/05_collective_communication.qmd`, anchor `fig-comm-compute-overlap`. The original PNG/SVG and owning section were reviewed before implementation.

Validation: `check.py` renders the existing actual chapter and verifies all sequence endpoints at desktop 1440 and phone 390 in both chapter themes, controls, pause/resume, reduced motion, Expand, and print/no-JavaScript fallback. Fresh motion contexts capture paused viewport screenshots independently of full component captures. Review artifacts are in `/tmp/original-v2-comm-compute-overlap-review/`. Production enrollment/build verification belongs to the integration wave.
