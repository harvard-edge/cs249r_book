# TinyTorch Changelog

All notable changes to the TinyTorch software package are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `v1.0.0`. Until then, minor version bumps may include
backwards-incompatible changes that are noted explicitly under "Breaking".

Releases are tagged in git as `tinytorch-vX.Y.Z` (the package-prefixed tag
distinguishes TinyTorch releases from book volume releases that share this
monorepo). See [GitHub Releases](https://github.com/harvard-edge/cs249r_book/releases?q=tinytorch)
for the canonical list.

## [Unreleased]

## [0.10.0] — 2026-04 (planned)

The first release tracked under `vX.Y.0` (rather than `v0.1.x`) to reflect
the package's maturity heading into the Volume II launch. This is *not* a
1.0 — TinyTorch is still pedagogical-first and APIs may change between
modules — but the jump from `0.1.x` to `0.10.x` signals a deliberate
broadening of the version space so that subsequent point releases have
room to breathe alongside the textbook's release cadence.

### Added
- **Licensing clarity.** TinyTorch is now distributed under MIT (replaces
  the leftover Apache-2.0 stub `LICENSE` file). A NOTICE block at the
  bottom of `LICENSE` documents the MIT-vs-CC-BY-NC-SA boundary between
  TinyTorch *software* and the surrounding *educational content*.
- **Explicit-version release path.** `tinytorch-publish-live` workflow now
  accepts an `explicit_version` input that bypasses the major/minor/patch
  auto-bump for non-incremental jumps (used for this 0.1.x → 0.10.0
  release). Ordinary releases continue to use `release_type`.
- **`settings.ini` covered by automated bumps.** The publish workflow
  previously updated `pyproject.toml`, `install.sh`, the Quarto
  announcement, and the README badge but skipped `settings.ini`, which
  silently drifted. The workflow now keeps it in sync with each release.

### Changed
- Version bumped from `0.1.9` → `0.10.0` in `pyproject.toml`,
  `settings.ini`, and the legacy site announcement banner.

### Notes for downstreams
- `tinytorch/__init__.py` reads its version from `pyproject.toml` at import
  time, so `import tinytorch; tinytorch.__version__` reflects the new
  number with no further changes.
- The MIT relicensing is a clarification, not a permission expansion: the
  pyproject metadata and README badge already declared MIT; only the
  `LICENSE` file text was wrong (it was an Apache-2.0 template stub with
  no copyright holder filled in, never actually granted to anyone).

## [0.1.9] and earlier

See [GitHub Releases](https://github.com/harvard-edge/cs249r_book/releases?q=tinytorch)
for pre-0.10 history. The 0.1.x line tracked early-access content
iteration during the Volume II writing process.
