# Reference data assets

Large numerical references are distributed as immutable GitHub Release
assets instead of being committed to Git history. The versioned asset index
is [`index.json`](index.json).

The accepted stability summaries and compact replay-file digests are pinned
in [`stability_regressions.json`](stability_regressions.json). Validate that
contract without downloading the large release assets:

```bash
python3 scripts/validate_stability_references.py
```

Updating an asset index entry does not update the accepted stability result.
Any intentional numerical change must update the stability lock separately
after the new bifurcation and stability diagrams have been reviewed.

The archive preserves paths below `data/reference/`. It contains accepted
KS1D bifurcation and stability projects, their configurations and logs, and
the eigensolver characterization snapshots.

Small inputs needed by normal CI remain in Git:

- `data/external/suitesparse/` contains the Matrix Market inputs;
- `source/models/KS_1D/tests/data/stability_replay/` contains compact KS1D
  stability states;
- `data/reference/eigensolvers/baseline_20260725/dat_A_lr_reference.csv`
  contains the expected spectrum used by the dense `A.dat` test.

## List available assets

```bash
python3 scripts/download_reference_data.py --list
```

## Download and extract the latest asset

```bash
python3 scripts/download_reference_data.py
```

The downloader verifies the file size and SHA-256 digest before extracting.
Existing identical files are retained. A differing local file is never
overwritten unless `--force` is specified.

## N=64 symmetric KS1D regression

The final `ks1d-n64-reference-cpu-omp` CI job runs only after all preliminary
tests pass. It uses one CPU thread and the unchanged numerical settings from
asset `20260730`, comparing nine branches by saved states rather than curve
IDs or point counts. Every saved physical residual and terminal outcome is
checked; restarting must leave every project file unchanged.
Each nonzero branch must match an archived discovery state after fixed-parameter
Newton correction (relative difference at most `1e-6`). Up to 201 reference
points per branch must also lie within `1e-3` of its fresh polyline in the
`(lambda, RMS)` plane, normalized by the reference parameter span and maximum
RMS norm. This tests branch extent without requiring identical sampling.

Run the same validation locally with a new work directory:

```bash
python3 scripts/download_reference_data.py --version 20260730 --destination build/ks1d_reference/asset
make KS1D_bd_cpu_omp test_KS1D_reference_cpu_omp.bin BUILD_DIR=build/ks1d_reference REAL_TYPE=double
python3 scripts/validate_ks1d_reference.py \
  --bd-executable build/ks1d_reference/KS1D_bd_cpu_omp.bin \
  --checker-executable build/ks1d_reference/test_KS1D_reference_cpu_omp.bin \
  --reference-dir build/ks1d_reference/asset/data/reference/KS1D/N64/symmetric \
  --work-dir build/ks1d_reference/replay
```

Logs, per-state checks, branch comparisons, and restart hashes are retained in
the work directory, including a `comparison.json` failure report.

## Build the release asset

```bash
python3 scripts/package_reference_data.py --version 20260730
```

The resulting archive is written below `build/reference_assets/`, which is
ignored by Git. After creating a new asset, update `index.json` with the
reported size and digest, upload the archive under a new immutable release
tag, and make that version the `latest` entry. Do not replace an asset that
has already been published.
