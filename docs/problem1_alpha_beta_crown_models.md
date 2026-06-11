# Problem 1. alpha-beta-CROWN Models Directory Exploration

This note summarizes the `complete_verifier/models` and `complete_verifier/exp_configs`
directories in the official alpha-beta-CROWN repository.

Inspection target:

- Repository: [Verified-Intelligence/alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- Models directory: [`complete_verifier/models`](https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/main/complete_verifier/models)
- Experiment configs: [`complete_verifier/exp_configs`](https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/main/complete_verifier/exp_configs)
- Documentation checked: [`complete_verifier/docs/abcrown_usage.md`](https://github.com/Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/abcrown_usage.md)
- Local inspection commit: `746b7d0128df1806c92381d1c8b3a66c9cba990c` (`June 2026 release`)

## 1. Models Directory

The `models` directory is not a single uniform model zoo. It is a collection of
small benchmark checkpoints, reference model definitions, and a few helper files
used by the example configurations.

File formats found in the checked-out directory:

| Format | Count | Meaning in this directory |
| --- | ---: | --- |
| `.pth` | 24 | PyTorch checkpoints for MNIST, CIFAR-10, CIFAR-100, ResNet, OVAL, ERAN, and related benchmarks |
| `.model` | 8 | Legacy benchmark model files used by SDP-style MNIST/CIFAR configs |
| `.pkl` | 4 | OVAL benchmark property/data files |
| `.py` | 4 | Model definitions or helper scripts, such as ResNet definitions and masked-model generation |
| `.md` | 2 | README files for model subsets |
| `.pt` | 1 | PyTorch checkpoint-style file used by a BaB-Attack model |
| no extension | 1 | Placeholder file, e.g. `UNZIP_MODELS_HERE` |

Subdirectories and examples:

| Subdirectory | Main contents |
| --- | --- |
| `eran` | MNIST fully connected networks, MNIST CNNs, and CIFAR CNN/ResNet checkpoints in `.pth` format |
| `sdp` | MNIST/CIFAR CNN benchmark models using `.model` files |
| `oval` | CIFAR base/deep/wide checkpoints and OVAL `.pkl` property files |
| `marabou_cifar10` | CIFAR-10 small/medium/large checkpoints originally aligned with Marabou-style benchmarks |
| `cifar10_resnet` | `resnet2b.pth`, `resnet4b.pth`, and Python ResNet definitions |
| `bab_attack` | CIFAR models for BaB-Attack experiments |
| `non_relu` | CIFAR models with Sigmoid activations, including a masked variant |
| `l2_norm` | CIFAR L2 robustness model |
| `custom_op` | Model checkpoint for custom operation examples |
| `custom_specs` | Small model used with a custom VNNLIB specification |
| `toy` | Tiny MNIST MLP checkpoint |
| `vnncomp22/cifar100` | CIFAR-100 model-definition helper and placeholder for external benchmark models |

Overall, the local model files are mostly PyTorch-style checkpoints and benchmark
artifacts. ONNX is supported by alpha-beta-CROWN, but the example ONNX models are
often referenced from external VNN-COMP benchmark repositories instead of being
stored directly in this `models` directory.

## 2. Verification Configuration Files

The main configuration directory is `complete_verifier/exp_configs`. In the
checked commit it contains 263 YAML files plus a few CSV, VNNLIB, README, and text
files.

Top-level config groups:

| Config group | YAML count | Purpose |
| --- | ---: | --- |
| `tutorial_examples` | 18 | Minimal examples for PyTorch, ONNX, VNNLIB, custom data loaders, custom model loaders, element-wise bounds, and custom specifications |
| `beta_crown` | 27 | Standard beta-CROWN examples for MNIST, CIFAR-10, CIFAR-100, OVAL, ResNet, and non-ReLU examples |
| `GCP-CROWN` | 13 | Examples that enable GCP-CROWN or cut-based strengthening for small CNN/MLP-style benchmarks |
| `BICCOS` | 128 | BICCOS/GCP/beta-CROWN comparison and ablation configurations |
| `bab_attack` | 6 | BaB-Attack experiment configurations plus additional property files |
| `vnncomp21` | 14 | VNN-COMP 2021 benchmark configurations such as ACASXu, ERAN, MNIST, CIFAR, OVAL, and NN4Sys |
| `vnncomp22` | 27 | VNN-COMP 2022 benchmark configurations, including CIFAR-100, TinyImageNet, RL/control, Carvana, and other benchmark families |
| `vnncomp23` | 13 | VNN-COMP 2023 benchmarks such as ViT, VGG, YOLO-related tasks, ML4ACOPF, GTSRB, and others |
| `vnncomp24` | 8 | VNN-COMP 2024 benchmark configurations |
| `vnncomp25` | 9 | VNN-COMP 2025 benchmark configurations such as TinyImageNet, Cora, CIFAR-100, and LSNC/ReLU-related tasks |

Common YAML sections:

| Section | Role |
| --- | --- |
| `general` | Global options such as device, seed, timeout/result behavior, complete verifier mode (`bab`, `mip`, `bab-refine`), and result output |
| `model` | Model source. PyTorch configs use `name` and `path`; ONNX configs use `onnx_path`; some configs also require `input_shape` |
| `data` | Dataset choice, normalization, start/end instance range, number of outputs, and custom dataloader hooks |
| `specification` | Verification property. It can define an Lp robustness ball with `norm` and `epsilon`, element-wise bounds, or a `vnnlib_path` |
| `solver` / `bab` / `attack` | Bound propagation, branch-and-bound, branching, attack, batch size, and solver tuning options |

The tutorial examples are especially useful templates. For example:

- `onnx_with_one_vnnlib.yaml` shows an ONNX model with a single VNNLIB property.
- `onnx_with_built-in_dataset_linf_bound.yaml` shows an ONNX model with a built-in dataset and Linf robustness radius.
- `pytorch_model_with_one_vnnlib.yaml` shows a PyTorch checkpoint plus a model definition and VNNLIB property.
- `custom_model_data_example.yaml` and related configs show how to plug in custom model/data loader functions.

## 3. How alpha-beta-CROWN Model Specification Differs From Marabou

In Assignment 3, the Marabou workflow was programmatic. The code loaded the ONNX
model with `Marabou.read_onnx`, obtained input and output variables, set lower and
upper bounds for each input feature, then added a linear output constraint such as
`y_target >= y_original` to search for a counterexample. A Marabou `UNSAT` result
meant no counterexample existed under those manually encoded constraints; `SAT`
meant a counterexample was found.

alpha-beta-CROWN is more configuration-centered. Instead of writing all constraints
through Python API calls, the usual workflow is to prepare a YAML file that names
the model, dataset or data source, input perturbation/specification, and solver
parameters. The same YAML structure can cover:

- PyTorch checkpoints plus Python model definitions.
- ONNX models via `model: onnx_path`.
- Built-in datasets such as MNIST/CIFAR or custom dataloaders.
- Lp robustness specifications such as Linf `epsilon`.
- Element-wise input bounds.
- General VNNLIB specifications, including CSV batches of many VNNLIB files.

This difference matters for Assignment 4. For our Wine MLP from Assignment 3, a
Marabou script directly encoded normalized feature bounds and output-class
comparison equations. In alpha-beta-CROWN, the equivalent experiment should be
expressed as an ONNX or PyTorch model plus a YAML/VNNLIB/custom-data setup that
represents the same normalized input region and the same class-margin property.

## 4. Takeaways For Our Assignment 4 Plan

The official examples suggest that the most relevant templates for the Wine MLP
experiment are the ONNX and custom-data tutorial configurations. Because the Wine
model has only 13 normalized input features, it is much smaller than the image
classification examples in the repository. The comparison with Marabou should
therefore focus on practical workflow differences, result status
(`verified`/`falsified`/`timeout` vs. `UNSAT`/`SAT`), and runtime on the same
epsilon sweep used in Assignment 3.
