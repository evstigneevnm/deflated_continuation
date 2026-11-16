"""Check the authoritative capability matrix against links and compiled metadata."""

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MATRIX = Path("source/contrib/nmfd-timesteppers/docs/capabilities.md")


def method_rows(text):
    section = text.split("## Orders And Controllers\n", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if cells[0] in ("Method", "---"):
            continue
        if len(cells) != 6 or cells[0] in rows:
            raise ValueError("Malformed or duplicate capability row")
        if not re.fullmatch(r"\d+|1-5", cells[1]) or not cells[2].isdigit():
            raise ValueError("Missing method order/embedding")
        if not re.fullmatch(r"\d+|current", cells[3]):
            raise ValueError("Missing dense order")
        if not set(cells[4].split(",")) <= {"F", "E", "B"}:
            raise ValueError("Unknown controller capability")
        if "](" not in cells[5]:
            raise ValueError("Method has no linked evidence")
        rows[cells[0]] = cells[1:4]
    if not rows:
        raise ValueError("Empty method matrix")
    return rows


def validate(root=ROOT, text=None, metadata=None):
    path = root / MATRIX
    text = path.read_text() if text is None else text
    rows = method_rows(text)
    for link in re.findall(r"\]\(([^)]+)\)", text):
        target = link.split("#", 1)[0]
        if target and not (path.parent / target).is_file():
            raise ValueError(f"Missing capability evidence: {target}")
    for heading in ("## Formulations", "## Backend And Application Evidence"):
        if heading not in text:
            raise ValueError(f"Missing capability section: {heading}")
    if metadata is not None:
        actual = {}
        for line in metadata.splitlines():
            name, *values = line.split("|")
            if len(values) != 3 or name in actual:
                raise ValueError("Malformed compiled capability metadata")
            actual[name] = values
        if rows != actual:
            raise ValueError(f"Documented orders differ from compiled tables: documented={rows}, compiled={actual}")
    return {"methods": len(rows), "links_checked": True, "compiled_metadata_checked": metadata is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-binary", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    metadata = None
    if args.metadata_binary:
        metadata = subprocess.run([str(args.metadata_binary.resolve())], check=True, text=True, capture_output=True).stdout
    result = validate(metadata=metadata)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
