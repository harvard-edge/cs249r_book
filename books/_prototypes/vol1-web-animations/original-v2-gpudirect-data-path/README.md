# Which copies disappear with GPUDirect RDMA?

The sending-node layout and paths from the original diagram. Compare two separate sends of the same symbolic n-byte payload; no time or throughput scale.

Host payload traffic disappears only in the direct case. NIC access is still a transfer over PCIe, followed by network transmission; no claim of zero transfer, one DMA transaction, or measured speedup.

Run `python3 check.py` for the disposable actual-chapter fixture and browser checks. The static SVG, caption, and manuscript remain unchanged. Motion is illustrative; endpoint states determine the visible ownership and values.

Source context: `books/vol2/03_network_fabrics/03_network_fabrics.qmd`, anchor `fig-gpudirect-data-path`. The original PNG/SVG and owning section were reviewed before implementation.

Validation: `check.py` renders the existing actual chapter and verifies all sequence endpoints at desktop 1440 and phone 390 in both chapter themes, controls, pause/resume, reduced motion, Expand, and print/no-JavaScript fallback. Fresh motion contexts capture paused viewport screenshots independently of full component captures. Review artifacts are in `/tmp/original-v2-gpudirect-data-path-review/`. Production enrollment/build verification belongs to the integration wave.
