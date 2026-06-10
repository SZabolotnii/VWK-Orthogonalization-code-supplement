#!/usr/bin/env python3
"""Run the required Paper 3a experiment/verification suite.

This driver is intentionally an audit wrapper, not a new scientific experiment.
It reruns the current moment-based finite-memory 3a gates and writes a compact
Markdown completion report. CF/3b production is out of scope.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPORT_DATE = "2026-06-08"
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "code" / "experiments" / "results"
LEAN_CACHE = Path(os.environ.get("VWK_LEAN_CACHE", str(ROOT / "vwk-lean" / ".lake" / "packages")))


@dataclass(frozen=True)
class CommandGate:
    gate: str
    description: str
    command: list[str]
    cwd: Path = ROOT


@dataclass(frozen=True)
class CommandResult:
    gate: str
    description: str
    command: list[str]
    cwd: Path
    returncode: int
    duration_s: float
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class ArtifactCheck:
    gate: str
    path: Path
    description: str


def paper_env() -> dict[str, str]:
    env = os.environ.copy()
    code_path = str(ROOT / "code")
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = code_path if not previous else f"{code_path}{os.pathsep}{previous}"
    return env


def ensure_lean_cache() -> None:
    lake_dir = ROOT / "vwk-lean" / ".lake"
    packages = lake_dir / "packages"
    if packages.exists():
        return
    if not LEAN_CACHE.exists():
        raise FileNotFoundError(f"Lean package cache not found: {LEAN_CACHE}")
    lake_dir.mkdir(parents=True, exist_ok=True)
    packages.symlink_to(LEAN_CACHE)


def command_string(command: list[str]) -> str:
    display = ["python" if part == sys.executable else part for part in command]
    return " ".join(display)


def run_gate(gate: CommandGate) -> CommandResult:
    started = time.perf_counter()
    completed = subprocess.run(
        gate.command,
        cwd=gate.cwd,
        env=paper_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CommandResult(
        gate=gate.gate,
        description=gate.description,
        command=gate.command,
        cwd=gate.cwd,
        returncode=completed.returncode,
        duration_s=time.perf_counter() - started,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def default_command_gates(*, include_lean: bool) -> list[CommandGate]:
    py = sys.executable
    gates = [
        CommandGate("G2", "package tests for basis and finite-memory estimator", [py, "-m", "pytest", "code/tests", "-q"]),
        CommandGate("G3", "Level-1 sanity checks", [py, "code/experiments/run_sanity.py"]),
        CommandGate(
            "G4",
            "one-lag synthetic H4 production grid",
            [py, "code/experiments/run_synthetic_h4.py", "--write"],
        ),
        CommandGate(
            "G7",
            "second-order memory=3 empirical-moment production grid",
            [py, "code/experiments/run_finite_memory_empirical.py", "--write"],
        ),
        CommandGate(
            "G8",
            "sample-efficiency production curve",
            [py, "code/experiments/run_sample_efficiency.py", "--write"],
        ),
        CommandGate(
            "G9",
            "real-world scalar diagnostic screen",
            [py, "code/experiments/run_realworld_screen.py", "--write"],
        ),
        CommandGate("A1", "discrete Askey verification", [py, "verification/verify_askey_discrete.py"]),
        CommandGate("A2", "continuous Askey verification", [py, "verification/verify_askey_continuous.py"]),
    ]
    if include_lean:
        gates.append(
            CommandGate(
                "G6",
                "Lean sorry-free fragment build",
                ["lake", "build", "VWK"],
                cwd=ROOT / "vwk-lean",
            )
        )
    return gates


def required_artifacts(*, include_lean: bool) -> list[ArtifactCheck]:
    checks = [
        ArtifactCheck("G1", ROOT / "theorem-proofs.md", "paper-level finite-memory proof layer"),
        ArtifactCheck("G5", ROOT / "manuscript-draft-uk.md", "Ukrainian 3a manuscript draft"),
        ArtifactCheck(
            "G4",
            RESULTS_DIR / f"synthetic_h4_production_summary_{REPORT_DATE}.csv",
            "synthetic H4 production summary",
        ),
        ArtifactCheck(
            "G4",
            RESULTS_DIR / f"synthetic_h4_production_replicates_{REPORT_DATE}.csv",
            "synthetic H4 production replicate table",
        ),
        ArtifactCheck(
            "G4",
            RESULTS_DIR / f"synthetic_h4_production_report_{REPORT_DATE}.md",
            "synthetic H4 production report",
        ),
        ArtifactCheck(
            "G7",
            RESULTS_DIR / f"finite_memory_empirical_production_summary_{REPORT_DATE}.csv",
            "finite-memory empirical production summary",
        ),
        ArtifactCheck(
            "G7",
            RESULTS_DIR / f"finite_memory_empirical_production_replicates_{REPORT_DATE}.csv",
            "finite-memory empirical production replicate table",
        ),
        ArtifactCheck(
            "G7",
            RESULTS_DIR / f"finite_memory_empirical_production_report_{REPORT_DATE}.md",
            "finite-memory empirical production report",
        ),
        ArtifactCheck(
            "G8",
            RESULTS_DIR / f"sample_efficiency_production_summary_{REPORT_DATE}.csv",
            "sample-efficiency production summary",
        ),
        ArtifactCheck(
            "G8",
            RESULTS_DIR / f"sample_efficiency_production_report_{REPORT_DATE}.md",
            "sample-efficiency production report",
        ),
        ArtifactCheck(
            "G9",
            RESULTS_DIR / f"realworld_screen_summary_{REPORT_DATE}.csv",
            "real-world screen summary",
        ),
        ArtifactCheck(
            "G9",
            RESULTS_DIR / f"realworld_screen_report_{REPORT_DATE}.md",
            "real-world screen report",
        ),
        ArtifactCheck("A1", ROOT / "verification" / "REPORT.txt", "stored discrete Askey report"),
        ArtifactCheck("A2", ROOT / "verification" / "REPORT_continuous.txt", "stored continuous Askey report"),
    ]
    if include_lean:
        checks.append(ArtifactCheck("G6", ROOT / "vwk-lean" / ".lake" / "packages", "local Lean/Mathlib cache link"))
    return checks


def tail(text: str, max_lines: int = 8) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.rstrip()]
    return "\n".join(lines[-max_lines:])


def write_report(path: Path, results: list[CommandResult], artifacts: list[ArtifactCheck], *, include_lean: bool) -> None:
    all_commands_pass = all(result.passed for result in results)
    missing_artifacts = [item for item in artifacts if not item.path.exists()]
    overall = all_commands_pass and not missing_artifacts
    lines = [
        "# Paper 3a Experiment Suite Audit",
        "",
        f"*Date: {REPORT_DATE} | scope: moment-based finite-memory 3a | Lean included: {include_lean}*",
        "",
        "## Verdict",
        "",
        "PASS" if overall else "FAIL",
        "",
        "This audit covers the current required 3a experiment and verification gates. CF/3b, full formal VWK verification, and real-world unbiased kernel recovery are explicitly outside this suite.",
        "",
        "## Command Gates",
        "",
        "| Gate | Status | Seconds | Command |",
        "|---|---|---:|---|",
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        rel_cwd = result.cwd.relative_to(ROOT) if result.cwd.is_relative_to(ROOT) else result.cwd
        lines.append(
            f"| {result.gate} | {status} | {result.duration_s:.1f} | "
            f"`(cd {rel_cwd} && {command_string(result.command)})` |"
        )
    lines.extend(["", "## Artifact Gates", "", "| Gate | Status | Artifact | Description |", "|---|---|---|---|"])
    for item in artifacts:
        status = "PASS" if item.path.exists() else "FAIL"
        rel_path = item.path.relative_to(ROOT) if item.path.is_relative_to(ROOT) else item.path
        lines.append(f"| {item.gate} | {status} | `{rel_path}` | {item.description} |")

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- The suite supports the finite-memory 3a claim: matched VWK projection versus misspecified Gaussian/Wiener projection under finite moments.",
            "- The synthetic evidence remains scoped to asymmetric finite-moment inputs; symmetric controls do not justify a symmetric non-Gaussian advantage claim.",
            "- The real-world screen is diagnostic only and does not establish unbiased Volterra-kernel recovery.",
            "- CF/moment-free 3b experiments are not necessary for the current 3a workstream and were not run.",
            "",
            "## Output Tails",
            "",
        ]
    )
    for result in results:
        lines.extend([f"### {result.gate} - {result.description}", ""])
        out_tail = tail(result.stdout)
        err_tail = tail(result.stderr)
        if out_tail:
            lines.extend(["```text", out_tail, "```", ""])
        if err_tail:
            lines.extend(["stderr:", "```text", err_tail, "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-lean", action="store_true", help="Include `lake build VWK` in the suite.")
    parser.add_argument("--no-write", action="store_true", help="Run the suite without writing the audit report.")
    parser.add_argument(
        "--out",
        default=f"code/experiments/results/experiment_suite_audit_{REPORT_DATE}.md",
        help="Audit report path relative to the Paper 3 root.",
    )
    args = parser.parse_args()

    if args.with_lean:
        ensure_lean_cache()

    results = [run_gate(gate) for gate in default_command_gates(include_lean=args.with_lean)]
    artifacts = required_artifacts(include_lean=args.with_lean)
    missing_artifacts = [item for item in artifacts if not item.path.exists()]
    overall = all(result.passed for result in results) and not missing_artifacts

    print("Paper 3a experiment suite audit")
    print(f"scope: moment-based finite-memory 3a | lean: {args.with_lean}")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.gate:<3} {status:<4} {result.duration_s:7.1f}s  {result.description}")
    for item in missing_artifacts:
        rel_path = item.path.relative_to(ROOT) if item.path.is_relative_to(ROOT) else item.path
        print(f"{item.gate:<3} FAIL artifact missing: {rel_path}")
    print("RESULT:", "PASS" if overall else "FAIL")

    if not args.no_write:
        report_path = ROOT / args.out
        write_report(report_path, results, artifacts, include_lean=args.with_lean)
        print(f"Wrote audit: {report_path.relative_to(ROOT)}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
