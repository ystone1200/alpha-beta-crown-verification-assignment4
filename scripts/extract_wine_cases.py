"""Extract Assignment 3 Wine MLP Marabou results into a compact baseline file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("models/wine_mlp_metadata.json"),
        help="Copied Wine MLP metadata file.",
    )
    parser.add_argument(
        "--marabou-results",
        type=Path,
        default=Path("assignment3/results/wine_marabou_results.json"),
        help="Assignment 3 Marabou result file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wine_verification_cases.json"),
        help="Compact baseline file to write.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_epsilon_result(epsilon_result: dict[str, Any]) -> dict[str, Any]:
    target_results = epsilon_result["target_results"]
    return {
        "epsilon": epsilon_result["epsilon"],
        "robust": epsilon_result["robust"],
        "status": "verified" if epsilon_result["robust"] else "falsified",
        "runtime_seconds": sum(item["runtime_seconds"] for item in target_results),
        "target_statuses": {
            str(item["target_class"]): item["status"] for item in target_results
        },
    }


def main() -> None:
    args = parse_args()
    metadata = load_json(args.metadata)
    marabou_results = load_json(args.marabou_results)

    output = {
        "dataset": metadata["dataset"],
        "architecture": metadata["architecture"],
        "model": "models/wine_mlp.onnx",
        "metadata": "models/wine_mlp_metadata.json",
        "input_space": "StandardScaler-normalized Wine feature space",
        "property": (
            "For a selected test sample x and radius epsilon, check whether "
            "there exists x' with |x'_i - x_i| <= epsilon such that any "
            "non-original logit is greater than or equal to the original logit."
        ),
        "epsilons": marabou_results["epsilons"],
        "class_names": marabou_results["class_names"],
        "samples": [],
    }

    for sample_result in marabou_results["samples"]:
        sample = sample_result["sample"]
        output["samples"].append(
            {
                "sample_id": sample["sample_id"],
                "test_index": sample["test_index"],
                "dataset_index": sample.get("dataset_index"),
                "true_class": sample["true_class"],
                "predicted_class": sample["predicted_class"],
                "logits": sample["logits"],
                "margin": sample["margin"],
                "original_input": sample["original_input"],
                "normalized_input": sample["normalized_input"],
                "marabou": [
                    summarize_epsilon_result(item)
                    for item in sample_result["epsilon_results"]
                ],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
