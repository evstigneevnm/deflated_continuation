# Reference data assets

Large numerical references are distributed as immutable GitHub Release
assets instead of being committed to Git history. The versioned asset index
is [`index.json`](index.json).

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

## Build the release asset

```bash
python3 scripts/package_reference_data.py --version 20260730
```

The resulting archive is written below `build/reference_assets/`, which is
ignored by Git. After creating a new asset, update `index.json` with the
reported size and digest, upload the archive under a new immutable release
tag, and make that version the `latest` entry. Do not replace an asset that
has already been published.
