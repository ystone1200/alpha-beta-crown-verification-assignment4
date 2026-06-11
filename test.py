"""Run alpha-beta-CROWN on the Wine MLP robustness properties."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


STATUS_MAP = {
    "unsat": "verified",
    "safe": "verified",
    "sat": "falsified",
    "unsafe": "falsified",
    "timeout": "timeout",
    "unknown": "timeout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abcrown-dir", type=Path, default=Path("alpha-beta-CROWN"))
    parser.add_argument("--config", type=Path, default=Path("configs/wine_mlp_base.yaml"))
    parser.add_argument("--cases", type=Path, default=Path("data/wine_verification_cases.json"))
    parser.add_argument("--spec-dir", type=Path, default=Path("specs/wine"))
    parser.add_argument("--results", type=Path, default=Path("results/wine_abcrown_results.json"))
    parser.add_argument("--sample-id", default="low_margin_3")
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--epsilons", type=float, nargs="*")
    parser.add_argument("--timeout", type=float, default=30)
    return parser.parse_args()


def epsilon_token(epsilon: float) -> str:
    return f"{epsilon:.3f}".replace(".", "_")


def find_python(abcrown_dir: Path) -> Path:
    candidate = abcrown_dir / ".venv" / "Scripts" / "python.exe"
    if not candidate.exists():
        raise SystemExit(
            f"Missing {candidate}. Install alpha-beta-CROWN first, e.g. "
            "`cd alpha-beta-CROWN && uv sync --python 3.11`."
        )
    return candidate


def normalize_status(raw_status: str) -> str:
    raw_status = raw_status.strip().lower()
    for key, value in STATUS_MAP.items():
        if raw_status.startswith(key):
            return value
    return "unknown"


def read_status(status_file: Path, stdout: str) -> tuple[str, str]:
    raw_status = ""
    if status_file.exists():
        raw_status = status_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    if not raw_status:
        match = re.findall(r"Result:\s*([A-Za-z0-9_-]+)", stdout)
        raw_status = match[-1] if match else "unknown"
    return raw_status, normalize_status(raw_status)


def load_cases(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `python scripts/extract_wine_cases.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def marabou_status_for(sample: dict[str, Any], epsilon: float) -> dict[str, Any]:
    for item in sample["marabou"]:
        if abs(float(item["epsilon"]) - epsilon) < 1e-12:
            return item
    raise KeyError(f"No Marabou result for {sample['sample_id']} epsilon={epsilon}")


def run_case(
    python: Path,
    abcrown_dir: Path,
    abcrown_script: Path,
    config: Path,
    spec_path: Path,
    status_file: Path,
    timeout: float,
) -> dict[str, Any]:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    if status_file.exists():
        status_file.unlink()

    command = [
        str(python.resolve()),
        str(abcrown_script.resolve()),
        "--config",
        str(config.resolve()),
        "--vnnlib_path",
        str(spec_path.resolve()),
        "--results_file",
        str(status_file.resolve()),
        "--timeout",
        str(timeout),
        "--device",
        "cpu",
    ]

    env = os.environ.copy()
    auto_lirpa_path = str((abcrown_dir / "auto_LiRPA").resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        auto_lirpa_path
        if not existing_pythonpath
        else auto_lirpa_path + os.pathsep + existing_pythonpath
    )

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=abcrown_script.parent,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    raw_status, status = read_status(status_file, completed.stdout)

    return {
        "status": status,
        "raw_status": raw_status,
        "runtime_seconds": elapsed,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-30:],
    }


def selected_samples(cases: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all_samples:
        return cases["samples"]
    for sample in cases["samples"]:
        if sample["sample_id"] == args.sample_id:
            return [sample]
    raise SystemExit(f"Unknown sample id: {args.sample_id}")


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    python = find_python(args.abcrown_dir)
    abcrown_script = args.abcrown_dir / "complete_verifier" / "abcrown.py"
    if not abcrown_script.exists():
        raise SystemExit(f"Missing alpha-beta-CROWN entry point: {abcrown_script}")

    epsilons = args.epsilons if args.epsilons else cases["epsilons"]
    results = {
        "model": cases["model"],
        "dataset": cases["dataset"],
        "input_space": cases["input_space"],
        "property": cases["property"],
        "abcrown_config": str(args.config),
        "abcrown_dir": str(args.abcrown_dir),
        "timeout_seconds": args.timeout,
        "samples": [],
    }

    for sample in selected_samples(cases, args):
        sample_output = {
            "sample_id": sample["sample_id"],
            "predicted_class": sample["predicted_class"],
            "margin": sample["margin"],
            "epsilon_results": [],
        }
        print(f"Verifying {sample['sample_id']} with alpha-beta-CROWN")
        for epsilon in epsilons:
            epsilon = float(epsilon)
            spec = args.spec_dir / f"{sample['sample_id']}_eps_{epsilon_token(epsilon)}.vnnlib"
            if not spec.exists():
                raise SystemExit(
                    f"Missing {spec}. Run `python scripts/generate_wine_specs.py` first."
                )
            status_file = Path("results") / "abcrown_status" / f"{spec.stem}.txt"
            abcrown_result = run_case(
                python=python,
                abcrown_dir=args.abcrown_dir,
                abcrown_script=abcrown_script,
                config=args.config,
                spec_path=spec,
                status_file=status_file,
                timeout=args.timeout,
            )
            marabou_result = marabou_status_for(sample, epsilon)
            row = {
                "epsilon": epsilon,
                "vnnlib": str(spec),
                "alpha_beta_crown": abcrown_result,
                "marabou": marabou_result,
                "matches_marabou": abcrown_result["status"] == marabou_result["status"],
            }
            sample_output["epsilon_results"].append(row)
            print(
                f"  eps={epsilon:g}: alpha-beta-CROWN={abcrown_result['status']} "
                f"({abcrown_result['raw_status']}, {abcrown_result['runtime_seconds']:.2f}s), "
                f"Marabou={marabou_result['status']}"
            )
        results["samples"].append(sample_output)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()
