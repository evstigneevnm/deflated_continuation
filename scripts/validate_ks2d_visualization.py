#!/usr/bin/env python3

import argparse
import ast
import json
import math
import struct
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate prepared KS2D NumPy fields and manifest metadata.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--minimum-records", type=int, default=1)
    parser.add_argument("--expected-backend")
    parser.add_argument("--maximum-absolute-value", type=float)
    return parser.parse_args()


def read_npy(path: Path):
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"Invalid NumPy magic: {path}")
        major, minor = stream.read(2)
        if (major, minor) != (1, 0):
            raise ValueError(f"Unsupported NumPy version {major}.{minor}: {path}")
        header_size = struct.unpack("<H", stream.read(2))[0]
        header = ast.literal_eval(stream.read(header_size).decode("latin1").strip())
        if header.get("fortran_order"):
            raise ValueError(f"Fortran-order NumPy arrays are unsupported: {path}")
        formats = {"<f4": ("<f", 4), "<f8": ("<d", 8)}
        if header.get("descr") not in formats:
            raise ValueError(f"Unsupported NumPy dtype {header.get('descr')}: {path}")
        value_format, value_size = formats[header["descr"]]
        shape = tuple(int(value) for value in header["shape"])
        count = math.prod(shape)
        payload = stream.read()
        if len(payload) != count * value_size:
            raise ValueError(f"NumPy payload size mismatch: {path}")
        values = [entry[0] for entry in struct.iter_unpack(value_format, payload)]
        return shape, values


def main():
    args = parse_args()
    manifest = args.manifest.resolve()
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) < args.minimum_records:
        raise ValueError(f"Expected at least {args.minimum_records} records, found {len(records)}")

    maximum = 0.0
    for record in records:
        if record.get("kind") != "physical_scalar_2d" or record.get("format") != "npy":
            raise ValueError(f"Unexpected visualization record kind or format: {record}")
        if record.get("shape") != [args.nx, args.ny]:
            raise ValueError(f"Unexpected manifest shape: {record.get('shape')}")
        if record.get("storage_order") != "last_index_fast":
            raise ValueError(f"Unexpected storage order: {record.get('storage_order')}")
        if args.expected_backend and record.get("producer_backend") != args.expected_backend:
            raise ValueError(f"Unexpected producer backend: {record.get('producer_backend')}")
        data_file = Path(record["data_file"])
        if not data_file.is_absolute():
            data_file = manifest.parent / data_file
        shape, values = read_npy(data_file)
        if shape != (args.nx, args.ny):
            raise ValueError(f"Unexpected NumPy shape {shape}: {data_file}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite field value: {data_file}")
        maximum = max(maximum, max((abs(value) for value in values), default=0.0))

    if args.maximum_absolute_value is not None and maximum > args.maximum_absolute_value:
        raise ValueError(
            f"Maximum absolute field value {maximum} exceeds {args.maximum_absolute_value}"
        )
    print(f"validated {len(records)} KS2D visualization records; max_abs={maximum:.17g}")


if __name__ == "__main__":
    main()
