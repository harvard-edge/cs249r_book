# Data Provenance (`15_verification`)

## `vnncomp2025_scored_benchmarks.csv`

- **Source**: Kaulen et al. (2025), *The 6th International Verification of Neural Networks Competition (VNN-COMP 2025): Summary and Results* (`@kaulen2025vnncomp`), arXiv:2512.19007, Table 3 ("Overview of all scored benchmarks").
- **Epistemic category**: `measured` (published competition benchmark inventory across all 25 scored benchmark suites: 16 regular-track suites and 9 extended-track suites).
- **Schema**: `category`, `benchmark`, `params_low`, `params_high` (parameter-count range per scored benchmark suite; single-network suites repeat the same count in both columns).
- **Accounting boundary**: Includes both regular-track suites (up to `cGAN` at 68,000,000 parameters) and extended-track suites (up to `VGGNet16` at 138,000,000 parameters), while excluding unscored suites omitted from Table 3.
