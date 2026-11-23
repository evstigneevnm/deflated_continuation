"""Run the branch writer and restarter in separate processes; never replace a run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


def check(executable, directory, negative):
    directory.mkdir(parents=True, exist_ok=False)

    def run(mode, root, **kwargs):
        return subprocess.run([str(executable.resolve()), mode, str(root)], **kwargs)

    run("--write", directory, check=True)
    files = {p: hashlib.sha256(p.read_bytes()).digest() for p in directory.glob("*/*.json")}
    cases = sorted(directory.glob("*/branch.json"))
    if len(cases) != 2:
        raise ValueError("Expected one- and four-section source archives")
    run("--restart", directory, check=True)
    for p, digest in files.items():
        if hashlib.sha256(p.read_bytes()).digest() != digest:
            raise ValueError(f"Restart changed {p}")
    for path in cases:
        before = json.loads(path.read_text())
        after = json.loads((path.parent / "restarted/branch.json").read_text())
        ref = before["interior_snapshot"]
        if after["restarted_from"] != dict(branch=before["id"], segment=ref["segment"], source_index=ref["source_index"]):
            raise ValueError("Restart provenance differs from the selected interior point")
        point, = (p for arm in before["segments"] for p in arm["points"] if p["source_index"] == ref["source_index"])
        if not point["verified"] or point["endpoint"] != "none" or not all(
            arm["points"][0]["parameter"] == point["parameter"] for arm in after["segments"]
        ):
            raise ValueError("Restart did not start at the verified interior orbit")

    if negative:
        # Change only disposable copies. The last case is structurally valid but
        # no longer satisfies the matching equations; it must not be Newton-corrected.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = cases[0].parent
            target = root / case.name
            shutil.copytree(case, target, ignore=shutil.ignore_patterns("restarted"))
            snapshot = target / "interior.json"
            original = snapshot.read_text()
            for defect, message in (("config", "Incompatible periodic orbit snapshot"),
                                    ("normal", "Invalid orbit snapshot section geometry"),
                                    ("coords", "Restored orbit failed verification")):
                data = json.loads(original)
                if defect == "config":
                    data["configuration"]["step"] *= 2
                elif defect == "normal":
                    data["nodes"][0]["normal"] = [0.] * data["dimension"]
                else:
                    data["nodes"][0]["coordinates"][0] += .1
                snapshot.write_text(json.dumps(data))
                attempt = run("--restart", root, capture_output=True, text=True)
                if not attempt.returncode or message not in attempt.stderr or (target / "restarted").exists():
                    raise ValueError(f"Did not cleanly reject {defect}: {attempt.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("new_directory", type=Path)
    parser.add_argument("--check-invalid", action="store_true", help="exercise the analytical-cycle corruption cases")
    args = parser.parse_args()
    check(args.executable, args.new_directory, args.check_invalid)
