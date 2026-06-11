"""Generate VNNLIB properties for the Wine MLP robustness checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/wine_verification_cases.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("specs/wine"),
    )
    return parser.parse_args()


def epsilon_token(epsilon: float) -> str:
    return f"{epsilon:.3f}".replace(".", "_")


def spec_path(output_dir: Path, sample_id: str, epsilon: float) -> Path:
    return output_dir / f"{sample_id}_eps_{epsilon_token(epsilon)}.vnnlib"


def render_vnnlib(sample: dict[str, Any], epsilon: float, class_names: list[str]) -> str:
    normalized_input = sample["normalized_input"]
    original_class = int(sample["predicted_class"])
    target_classes = [
        index for index in range(len(class_names)) if index != original_class
    ]

    lines: list[str] = [
        f"; Wine MLP local robustness property for {sample['sample_id']}",
        f"; epsilon = {epsilon} in StandardScaler-normalized feature space",
        "; Unsafe condition: some target logit is >= the original predicted logit.",
        "",
    ]

    for index in range(len(normalized_input)):
        lines.append(f"(declare-const X_{index} Real)")
    lines.append("")
    for index in range(len(class_names)):
        lines.append(f"(declare-const Y_{index} Real)")
    lines.append("")

    for index, value in enumerate(normalized_input):
        lower = value - epsilon
        upper = value + epsilon
        lines.append(f"(assert (>= X_{index} {lower:.17g}))")
        lines.append(f"(assert (<= X_{index} {upper:.17g}))")
    lines.append("")

    lines.append("(assert (or")
    for target_class in target_classes:
        # alpha-beta-CROWN proves the VNNLIB unsafe region is unreachable.
        # y_original <= y_target means the class can change or tie.
        lines.append(f"    (and (<= Y_{original_class} Y_{target_class}))")
    lines.append("))")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for sample in cases["samples"]:
        for epsilon in cases["epsilons"]:
            path = spec_path(args.output_dir, sample["sample_id"], float(epsilon))
            path.write_text(
                render_vnnlib(sample, float(epsilon), cases["class_names"]),
                encoding="utf-8",
            )
            written.append(path)

    print(f"Wrote {len(written)} VNNLIB files to {args.output_dir}")


if __name__ == "__main__":
    main()
