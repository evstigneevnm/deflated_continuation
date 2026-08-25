#!/usr/bin/env python3
"""Replay accepted nontrivial full-KS1D stability states and transitions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys


STATE_RESULT = re.compile(
    r"Single-state stability result: status=([^,]+), "
    r"unstable=\((-?\d+),(-?\d+)\)"
)
TRANSITION_RESULT = re.compile(
    r"Two-state stability transition result: status=([^,]+), "
    r"lambda=([^,]+), before=\((-?\d+),(-?\d+)\), "
    r"after=\((-?\d+),(-?\d+)\), iterations=(\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--manifest",
        default=(
            "source/models/KS_1D/tests/data/"
            "stability_replay/manifest.json"
        ),
    )
    parser.add_argument(
        "--device",
        help="Optional SCFD device selector, such as auto",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported stability replay manifest")
    return manifest


def validate_fixture(path: Path, expected_size: int) -> None:
    values = [
        float(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != expected_size:
        raise RuntimeError(
            f"{path}: expected {expected_size} values, found {len(values)}"
        )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{path}: fixture contains a non-finite value")


def transition_state_ids(
    manifest: dict,
    known_state_ids: set[str],
) -> set[str]:
    referenced: set[str] = set()
    for transition in manifest["transitions"]:
        for field in ("first_state", "second_state"):
            state_id = transition.get(field)
            if state_id not in known_state_ids:
                raise RuntimeError(
                    f"{transition.get('id', 'transition')}: unknown "
                    f"{field} {state_id!r}"
                )
            referenced.add(state_id)
    return referenced


def run_replay(
    executable: Path,
    config: Path,
    arguments: list[str],
    device: str | None,
    timeout: float,
) -> str:
    command = [
        str(executable),
        str(config),
        *arguments,
        "--confirm",
        "--quiet",
    ]
    if device is not None:
        command.append(device)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"replay command failed with rc={completed.returncode}:\n"
            f"{output}"
        )
    return output


def require_equal(actual: tuple[int, int], expected: list[int], label: str) -> None:
    expected_pair = (int(expected[0]), int(expected[1]))
    if actual != expected_pair:
        raise RuntimeError(
            f"{label}: expected signature {expected_pair}, got {actual}"
        )


def main() -> int:
    args = parse_args()
    executable = Path(args.executable).resolve()
    config = Path(args.config).resolve()
    manifest_path = Path(args.manifest).resolve()
    fixture_directory = manifest_path.parent
    manifest = load_manifest(manifest_path)

    states: dict[str, dict] = {}
    for state in manifest["states"]:
        state_path = fixture_directory / state["file"]
        validate_fixture(state_path, int(manifest["state_size"]))
        state = dict(state)
        state["path"] = state_path
        states[state["id"]] = state

    transition_states = transition_state_ids(manifest, set(states))
    validated_states: set[str] = set()
    commands = 0
    for state_id, state in states.items():
        if state_id in transition_states:
            continue
        output = run_replay(
            executable,
            config,
            [
                "--state-file",
                str(state["path"]),
                "--parameter",
                repr(float(state["parameter"])),
            ],
            args.device,
            args.timeout,
        )
        match = STATE_RESULT.search(output)
        if match is None:
            raise RuntimeError(
                f"{state['id']}: missing machine-readable result:\n{output}"
            )
        if match.group(1) != "complete":
            raise RuntimeError(
                f"{state['id']}: classification status is {match.group(1)}"
            )
        require_equal(
            (int(match.group(2)), int(match.group(3))),
            state["unstable"],
            state["id"],
        )
        validated_states.add(state_id)
        commands += 1

    transitions_checked = 0
    for transition in manifest["transitions"]:
        first = states[transition["first_state"]]
        second = states[transition["second_state"]]
        output = run_replay(
            executable,
            config,
            [
                "--state-file",
                str(first["path"]),
                "--parameter",
                repr(float(first["parameter"])),
                "--state-file-2",
                str(second["path"]),
                "--parameter-2",
                repr(float(second["parameter"])),
            ],
            args.device,
            args.timeout,
        )
        match = TRANSITION_RESULT.search(output)
        if match is None:
            raise RuntimeError(
                f"{transition['id']}: missing transition result:\n{output}"
            )
        if match.group(1) != "success":
            raise RuntimeError(
                f"{transition['id']}: transition status is {match.group(1)}"
            )
        parameter = float(match.group(2))
        parameter_error = abs(parameter - float(transition["parameter"]))
        if parameter_error > float(transition["parameter_tolerance"]):
            raise RuntimeError(
                f"{transition['id']}: refined parameter error "
                f"{parameter_error:.3e} exceeds tolerance "
                f"{float(transition['parameter_tolerance']):.3e}"
            )
        require_equal(
            (int(match.group(3)), int(match.group(4))),
            transition["before"],
            f"{transition['id']} before",
        )
        require_equal(
            (int(match.group(5)), int(match.group(6))),
            transition["after"],
            f"{transition['id']} after",
        )
        validated_states.update(
            (transition["first_state"], transition["second_state"])
        )
        transitions_checked += 1
        commands += 1

    missing_states = set(states) - validated_states
    if missing_states:
        raise RuntimeError(
            "states were not validated by a standalone or transition "
            f"replay: {sorted(missing_states)}"
        )

    backend = args.device if args.device is not None else "host"
    checks = len(validated_states) + transitions_checked
    print(
        "KS1D full stability replay PASSED: "
        f"backend={backend}, checks={checks}, commands={commands}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
