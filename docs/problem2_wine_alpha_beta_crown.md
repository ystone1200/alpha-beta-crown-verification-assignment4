# Problem 2. Running alpha-beta-CROWN on an External Model

## Model and Dataset

I reused the Wine MLP model from Assignment 3 as the external model for
alpha-beta-CROWN.

| Item | Choice |
| --- | --- |
| Dataset | `sklearn.datasets.load_wine()` |
| Input space | 13 StandardScaler-normalized Wine features |
| Model format | ONNX |
| Architecture | `13 -> 16 -> ReLU -> 8 -> ReLU -> 3` |
| Test sample | `low_margin_3` |
| Original predicted class | `class_1` |
| Logit margin | `4.312880516052246` |

This model is external to the official alpha-beta-CROWN `models` directory.

## Verification Property

The property matches the Assignment 3 Marabou setup. For each epsilon, the input
box is:

```text
x_i - epsilon <= x'_i <= x_i + epsilon
```

The VNNLIB file encodes the unsafe condition:

```text
y_target >= y_original for some target class
```

Therefore:

- alpha-beta-CROWN `unsat` means the unsafe condition is unreachable, so the
  sample is verified robust.
- alpha-beta-CROWN `sat` means a counterexample exists, so the property is
  falsified.

## Results

Command:

```bash
python test.py --sample-id low_margin_3 --timeout 30
```

Results are saved in `results/wine_abcrown_results.json`.

| Epsilon | alpha-beta-CROWN | alpha-beta-CROWN internal time (s) | Wrapper wall time (s) | Marabou baseline | Marabou time (s) |
| ---: | --- | ---: | ---: | --- | ---: |
| 0.01 | verified (`unsat`) | 0.193 | 5.577 | verified | 0.000846 |
| 0.05 | verified (`unsat`) | 0.185 | 5.352 | verified | 0.000873 |
| 0.1 | verified (`unsat`) | 0.185 | 5.446 | verified | 0.001280 |
| 0.3 | falsified (`sat`) | 0.027 | 5.318 | falsified | 0.003225 |
| 0.5 | falsified (`sat`) | 0.033 | 5.270 | falsified | 0.003081 |
| 1.0 | falsified (`sat`) | 0.026 | 5.042 | falsified | 0.003118 |

All six alpha-beta-CROWN statuses matched the Marabou baseline. For epsilon
`0.01`, `0.05`, and `0.1`, alpha-beta-CROWN verified the property using initial
CROWN bounds. For epsilon `0.3`, `0.5`, and `1.0`, the PGD attack stage found
counterexamples quickly.

## Interpretation

The robust/non-robust boundary is the same as in Assignment 3: the selected Wine
sample is robust up to the tested radius `0.1`, but not robust at `0.3` or above.
The main workflow difference is that Marabou encoded the input bounds and output
comparison directly in Python, while alpha-beta-CROWN used a YAML configuration
plus VNNLIB property files. The YAML/VNNLIB setup is more structured and easier
to batch once the files exist, but it required more environment and path setup.
