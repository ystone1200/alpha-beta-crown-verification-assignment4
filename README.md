# Assignment 4: alpha-beta-CROWN Wine MLP Verification

This repository verifies an external Wine dataset MLP with alpha-beta-CROWN and
compares the result with the Marabou experiment from Assignment 3.

## Files

- `docs/problem1_alpha_beta_crown_models.md`: exploration of the official
  alpha-beta-CROWN models and configuration directories.
- `models/wine_mlp.onnx`: external ONNX model reused from Assignment 3.
- `models/wine_mlp_metadata.json`: model architecture, scaler, and dataset metadata.
- `data/wine_verification_cases.json`: selected Wine test samples and Marabou baseline results.
- `specs/wine/*.vnnlib`: VNNLIB unsafe-region specifications for each sample and epsilon.
- `configs/wine_mlp_base.yaml`: alpha-beta-CROWN base configuration.
- `test.py`: runs alpha-beta-CROWN on the Wine MLP VNNLIB properties.
- `results/wine_abcrown_results.json`: recorded alpha-beta-CROWN results.

## Environment

alpha-beta-CROWN is installed as a local tool directory and is intentionally not
committed to this repository.

```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
uv sync --python 3.11
cd ..
```

The run used alpha-beta-CROWN commit `746b7d0128df1806c92381d1c8b3a66c9cba990c`.
On this Windows setup, `test.py` prepends `alpha-beta-CROWN/auto_LiRPA` to
`PYTHONPATH` so the verifier imports the submodule package correctly.

## Run

Generate VNNLIB files again if needed:

```bash
python scripts/generate_wine_specs.py
```

Run the default comparison sample:

```bash
python test.py
```

The default run verifies `low_margin_3` for epsilon values
`0.01, 0.05, 0.1, 0.3, 0.5, 1.0` in the normalized Wine feature space and writes
`results/wine_abcrown_results.json`.

Run every stored sample:

```bash
python test.py --all-samples
```

## Verification Property

For a selected normalized test input `x` and radius `epsilon`, the VNNLIB files
encode the unsafe condition:

```text
exists x' such that |x'_i - x_i| <= epsilon
and some non-original class logit >= the original predicted class logit
```

alpha-beta-CROWN returns `unsat` when this unsafe region is unreachable, which
means the sample is verified robust for that epsilon. It returns `sat` when it
finds a counterexample.
