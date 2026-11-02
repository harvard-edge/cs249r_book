# How does the same index reach different storage offsets?

Perform each indexed read against both original alternative layouts; preserve the backing arrays while the returned value travels to the read result.

Scope: The source 2 × 3 tensor and both alternative serializations. Each cell is one scalar element; offsets count elements. No byte width or cache model is assumed. The logical tensor is a view of either alternative storage, not a third allocation.

Guard: No tensor copy, cache misses, byte counts, or speedup is inferred from an index mapping.

Run `/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-tensor-memory-layout/check.py`. Fixture uses the complete existing chapter; production enrollment and static builds remain orchestration work.
