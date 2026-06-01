#!/usr/bin/env python3
"""Run a small GitHub Actions-style workflow locally.

The runner intentionally supports the subset needed by this repository:
workflow-level env/defaults, jobs, and run steps. PyYAML is used when
available; otherwise a small fallback parser handles the local workflow file.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_WORKFLOW = ".github/workflows/operations.yml"


@dataclass
class JobResult:
    name: str
    status: str
    duration: float
    failed_step: str | None = None
    returncode: int = 0
    message: str | None = None


def strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for i, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()


def scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value in ("{}",):
        return {}
    if value in ("[]",):
        return []
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none", "~"):
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    return value


def preprocess_yaml(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        clean = strip_comment(raw)
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        lines.append((indent, clean[indent:]))
    return lines


def parse_block_scalar(lines: list[tuple[int, str]], index: int, parent_indent: int, folded: bool) -> tuple[str, int]:
    collected: list[str] = []
    block_indent: int | None = None
    while index < len(lines):
        indent, content = lines[index]
        if indent <= parent_indent:
            break
        if block_indent is None:
            block_indent = indent
        drop = min(indent, block_indent)
        collected.append(" " * (indent - drop) + content)
        index += 1
    if folded:
        return " ".join(part.strip() for part in collected), index
    return "\n".join(collected), index


def parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    current_indent, content = lines[index]
    if current_indent < indent:
        return {}, index
    if content.startswith("- "):
        return parse_yaml_list(lines, index, current_indent)
    return parse_yaml_dict(lines, index, current_indent)


def parse_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"expected key/value YAML line, got: {text}")
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def parse_value(lines: list[tuple[int, str]], index: int, indent: int, value: str) -> tuple[Any, int]:
    if value in ("|", ">"):
        return parse_block_scalar(lines, index, indent, folded=(value == ">"))
    if value:
        return scalar(value), index
    if index >= len(lines) or lines[index][0] <= indent:
        return {}, index
    return parse_yaml_block(lines, index, lines[index][0])


def parse_yaml_dict(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"unexpected indentation before: {content}")
        if content.startswith("- "):
            break
        key, value = parse_key_value(content)
        index += 1
        result[key], index = parse_value(lines, index, line_indent, value)
    return result, index


def parse_yaml_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"unexpected indentation before: {content}")
        if not content.startswith("- "):
            break
        item_text = content[2:].strip()
        index += 1
        if not item_text:
            item, index = parse_yaml_block(lines, index, lines[index][0])
        elif ":" in item_text:
            key, value = parse_key_value(item_text)
            item = {}
            item[key], index = parse_value(lines, index, line_indent, value)
            if index < len(lines) and lines[index][0] > line_indent:
                extra, index = parse_yaml_dict(lines, index, lines[index][0])
                item.update(extra)
        else:
            item = scalar(item_text)
        result.append(item)
    return result, index


def fallback_yaml_load(text: str) -> dict[str, Any]:
    lines = preprocess_yaml(text)
    data, index = parse_yaml_block(lines, 0, 0)
    if index != len(lines):
        raise ValueError("YAML fallback parser did not consume the full file")
    if not isinstance(data, dict):
        raise ValueError("workflow root must be a mapping")
    return data


def load_workflow(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return fallback_yaml_load(text)

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a workflow mapping")
    return data


def as_str_env(env: dict[str, Any]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in env.items()}


def merge_env(*envs: dict[str, Any] | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    for env in envs:
        if env:
            merged.update(as_str_env(env))
    return merged


def get_run_defaults(workflow: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    workflow_run = workflow.get("defaults", {}).get("run", {})
    job_run = job.get("defaults", {}).get("run", {})
    if isinstance(workflow_run, dict):
        result.update(workflow_run)
    if isinstance(job_run, dict):
        result.update(job_run)
    return result


def shell_executable(shell: str) -> str | None:
    if shell.startswith("bash"):
        return "/bin/bash"
    if shell.startswith("sh"):
        return "/bin/sh"
    return None


def command_for_shell(command: str, shell: str) -> str:
    if shell.startswith("bash"):
        return "set -eo pipefail\n" + command
    if shell.startswith("sh"):
        return "set -e\n" + command
    return command


def expand_expressions(value: str, env: dict[str, str], project_root: Path) -> str:
    def replacement(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        if expression.startswith("env."):
            return env.get(expression[4:], "")
        if expression == "github.workspace":
            return str(project_root)
        return match.group(0)

    return re.sub(r"\$\{\{\s*([^}]+?)\s*\}\}", replacement, value)


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def eval_condition(condition: Any, env: dict[str, str]) -> bool:
    if condition is None:
        return True
    if isinstance(condition, bool):
        return condition
    text = str(condition).strip()
    if text in ("", "true", "${{ true }}"):
        return True
    if text in ("false", "${{ false }}"):
        return False
    match = re.fullmatch(r"\$\{\{\s*env\.([A-Za-z_][A-Za-z0-9_]*)\s*([!=]=)\s*'([^']*)'\s*\}\}", text)
    if match:
        name, op, expected = match.groups()
        actual = env.get(name, "")
        return actual == expected if op == "==" else actual != expected
    print(f"warning: unsupported if condition {text!r}; running it", file=sys.stderr)
    return True


def resolve_workdir(project_root: Path, default_workdir: str | None, step: dict[str, Any], env: dict[str, str]) -> Path:
    workdir = step.get("working-directory") or default_workdir or "."
    workdir = expand_expressions(str(workdir), env, project_root)
    path = Path(str(workdir))
    if not path.is_absolute():
        path = project_root / path
    return path


def run_job(
    project_root: Path,
    workflow: dict[str, Any],
    name: str,
    job: dict[str, Any],
    extra_env: dict[str, str],
    timeout: float | None,
    dry_run: bool,
) -> JobResult:
    start = time.monotonic()
    workflow_env = workflow.get("env", {}) if isinstance(workflow.get("env", {}), dict) else {}
    job_env = job.get("env", {}) if isinstance(job.get("env", {}), dict) else {}
    env = os.environ.copy()
    env.update(merge_env(workflow_env, job_env, extra_env))

    if not eval_condition(job.get("if"), env):
        return JobResult(name=name, status="SKIP", duration=time.monotonic() - start)

    defaults = get_run_defaults(workflow, job)
    default_shell = str(defaults.get("shell", "bash"))
    default_workdir = defaults.get("working-directory")
    steps = job.get("steps", [])
    if not isinstance(steps, list):
        return JobResult(name=name, status="FAIL", duration=time.monotonic() - start, failed_step="steps", returncode=2)

    print(f"\n==> job: {name}", flush=True)
    for index, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, dict):
            continue
        step = raw_step
        step_name = str(step.get("name", f"step {index}"))
        step_env = step.get("env", {}) if isinstance(step.get("env", {}), dict) else {}
        run_env = env.copy()
        run_env.update(as_str_env(step_env))

        if not eval_condition(step.get("if"), run_env):
            print(f"  - skip: {step_name}", flush=True)
            continue
        if "uses" in step:
            print(f"  - skip uses step locally: {step_name} ({step['uses']})", flush=True)
            continue
        raw_command = step.get("run")
        if raw_command is None:
            continue
        command = expand_expressions(str(raw_command), run_env, project_root)

        shell = str(step.get("shell", default_shell))
        cwd = resolve_workdir(project_root, None if default_workdir is None else str(default_workdir), step, run_env)
        print(f"  - run: {step_name}", flush=True)
        print(f"    cwd: {display_path(cwd, project_root)}", flush=True)
        print(f"    $ {command}", flush=True)
        if dry_run:
            continue

        try:
            completed = subprocess.run(
                command_for_shell(str(command), shell),
                cwd=str(cwd),
                env=run_env,
                shell=True,
                executable=shell_executable(shell),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return JobResult(
                name=name,
                status="FAIL",
                duration=time.monotonic() - start,
                failed_step=step_name,
                returncode=124,
                message=f"timeout after {timeout:g}s" if timeout is not None else "timeout",
            )
        except OSError as exc:
            return JobResult(
                name=name,
                status="FAIL",
                duration=time.monotonic() - start,
                failed_step=step_name,
                returncode=127,
                message=str(exc),
            )
        if completed.returncode != 0:
            return JobResult(
                name=name,
                status="FAIL",
                duration=time.monotonic() - start,
                failed_step=step_name,
                returncode=completed.returncode,
            )

    return JobResult(name=name, status="PASS", duration=time.monotonic() - start)


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--env expects KEY=VALUE, got {value!r}")
        key, val = value.split("=", 1)
        result[key] = val
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a GitHub Actions-style workflow locally.")
    parser.add_argument("workflow", nargs="?", default=DEFAULT_WORKFLOW, help=f"workflow YAML path (default: {DEFAULT_WORKFLOW})")
    parser.add_argument("--job", action="append", default=[], help="run only the named job; may be passed multiple times")
    parser.add_argument("--list", action="store_true", help="list jobs and exit")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed job")
    parser.add_argument("--timeout", type=float, default=None, help="timeout in seconds for each run step")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    parser.add_argument("--project-root", default=None, help="project root directory (default: script directory)")
    parser.add_argument("--env", action="append", default=[], help="override workflow env, KEY=VALUE")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_root
    workflow_path = Path(args.workflow)
    if not workflow_path.is_absolute():
        workflow_path = project_root / workflow_path

    workflow = load_workflow(workflow_path)
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        raise SystemExit(f"{workflow_path} does not define a jobs mapping")

    selected = set(args.job)
    if args.list:
        for name in jobs:
            print(name)
        return 0

    extra_env = parse_env_overrides(args.env)
    results: list[JobResult] = []
    for name, job in jobs.items():
        if selected and name not in selected:
            continue
        if not isinstance(job, dict):
            results.append(JobResult(name=name, status="FAIL", duration=0.0, failed_step="job definition", returncode=2))
            continue
        result = run_job(project_root, workflow, name, job, extra_env, args.timeout, args.dry_run)
        results.append(result)
        if args.fail_fast and result.status == "FAIL":
            break

    if selected:
        missing = selected.difference(jobs.keys())
        for name in sorted(missing):
            results.append(JobResult(name=name, status="FAIL", duration=0.0, failed_step="unknown job", returncode=2))

    print("\nSummary")
    failed = 0
    for result in results:
        detail = ""
        if result.failed_step:
            detail = f" ({result.failed_step}, rc={result.returncode})"
        if result.message:
            detail += f" {result.message}"
        print(f"  {result.status:4} {result.name:35} {result.duration:8.2f}s{detail}")
        failed += int(result.status == "FAIL")

    if failed:
        print(f"\nFailed jobs: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
