# How does each expert result find its original token?

The original sixteen tokens a–p, four ranks, and one expert per rank. Top-1 routing; prime marks an expert output, not a numerical value. One forward pass; backward exchanges are outside this sequence.

Exact original token-to-expert assignment and inverse permutation. Prime denotes expert transformation; output returns to token owner. Top-1 only; no fabricated numeric activation, bandwidth, or load claim.

Run `python3 check.py` for the disposable actual-chapter fixture and browser checks. The static SVG, caption, and manuscript remain unchanged. Motion is illustrative; endpoint states determine the visible ownership and values.

Source context: `books/vol2/06_distributed_training/06_distributed_training.qmd`, anchor `fig-moe-all-to-all-routing`. The original PNG/SVG and owning section were reviewed before implementation.

Validation: `check.py` renders the existing actual chapter and verifies all sequence endpoints at desktop 1440 and phone 390 in both chapter themes, controls, pause/resume, reduced motion, Expand, and print/no-JavaScript fallback. Fresh motion contexts capture paused viewport screenshots independently of full component captures. Review artifacts are in `/tmp/original-v2-moe-all-to-all-routing-review/`. Production enrollment/build verification belongs to the integration wave.
