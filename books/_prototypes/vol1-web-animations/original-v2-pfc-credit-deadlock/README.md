# What must change before a deadlocked queue can forward?

The original four-switch cycle, with four declared toy packet slots per queue. For this toy only, full queues pause their predecessor and any free slot allows resume. A watchdog drops one packet; real thresholds and recovery policies vary.

PAUSE direction is opposite to the packet dependency. No queue drains while all are full. Watchdog drop is recorded as loss; resumption is conditional on the explicitly simplified threshold rule.

Run `python3 check.py` for the disposable actual-chapter fixture and browser checks. The static SVG, caption, and manuscript remain unchanged. Motion is illustrative; endpoint states determine the visible ownership and values.

Source context: `books/vol2/03_network_fabrics/03_network_fabrics.qmd`, anchor `fig-pfc-credit-deadlock`. The original PNG/SVG and owning section were reviewed before implementation.

Validation: `check.py` renders the existing actual chapter and verifies all sequence endpoints at desktop 1440 and phone 390 in both chapter themes, controls, pause/resume, reduced motion, Expand, and print/no-JavaScript fallback. Fresh motion contexts capture paused viewport screenshots independently of full component captures. Review artifacts are in `/tmp/original-v2-pfc-credit-deadlock-review/`. Production enrollment/build verification belongs to the integration wave.
