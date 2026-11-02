# Which input values are copied into each unfolded row?

Materialize each of the four exact source patches, retaining input identity and making overlapping copies visible; then read each completed row with the stacked weights.

Scope: Exact source values: channels 1–9 and 10–18; two 2 × 2 kernels stack into weights 1–8. Valid stride-one lowering creates four rows of eight values. This version materializes the buffer; implicit GEMM need not.

Guard: Exact two-channel source case; outputs are explicitly supporting calculations. No implicit-GEMM allocation claim or performance verdict.

Run `/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-im2col-diagram/check.py`. Fixture uses the complete existing chapter; production enrollment and static builds remain orchestration work.
