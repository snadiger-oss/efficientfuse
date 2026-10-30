# GreenFuse

GreenFuse is a CSE 585 research project that tests whether different ways of grouping GPU operations, with or without CUDA Graph replay, can reduce GPU energy while staying within a runtime limit.

The GitHub repository is **efficientfuse**. The Python package and command are named **greenfuse**.

**Current status:** the initial implementation covers all four project roles for workload W1. Real CPU compilation, correctness, timing, and executor reuse have been checked. Selection and cache logic have been tested using clearly labeled synthetic data. **NVIDIA GPU execution, real energy measurements, and the full GPU calibration/cache demonstration still need hardware validation.** The local test run on October 29, 2026 completed 117 tests: 110 passed and seven GPU tests were skipped.

## What the project runs

The first workload is bias addition, ReLU, and residual addition:

```python
import torch


def workload_w1(x, bias, residual):
    return torch.relu(x + bias) + residual
```

The default inputs are `x` and `residual` with shape `[32, 256]`, and `bias` with shape `[256]`. Inputs are contiguous FP32 tensors and execution is forward-only.

The planner automatically constructs these four operation partitions:

| Partition | Requested operation groups |
| --- | --- |
| `p00` | `[add → ReLU → add]` |
| `p01` | `[add → ReLU] [add]` |
| `p10` | `[add] [ReLU → add]` |
| `p11` | `[add] [ReLU] [add]` |

Each partition has ordinary execution (`launch`) and explicit CUDA Graph replay (`graph`), giving **eight requested variants**. Requested groups are inputs to the compiler; they do not guarantee a particular number of GPU kernels. Kernel counts are collected separately from profiler evidence.

The complete workflow is:

1. Capture the supported operation graph and enumerate partitions.
2. Compile each requested group independently with TorchInductor.
3. Check outputs against the original workload using original and changed input values.
4. Measure runtime and GPU energy in separate repeated passes.
5. Keep candidates whose median runtime is at most `1.05 × fastest eligible median runtime`.
6. Choose the feasible candidate with the lowest median energy. Break ties by runtime, then candidate ID.
7. Repeat the full experiment with fresh measurements. Save the decision only if validation succeeds.
8. Rebuild the selected executor when necessary and reuse it.

The original uncompiled PyTorch workload is measured as an external baseline. It is excluded from the compiled-candidate selection set. Partial grouping is a hypothesis to test; an energy advantage is not guaranteed.

## Fresh installation on a GPU machine

Follow these steps in order. The concrete installation below targets **Ubuntu 22.04/24.04 on Linux x86_64**, using the distribution's Python 3.10 or 3.12 and an NVIDIA GPU compatible with the selected PyTorch build.

Run GPU checks and experiments on an allocated GPU node. On a managed cluster, the administrator may provide the system tools through modules or an environment image instead of allowing `sudo`.

### 1. Check the NVIDIA driver

```bash
nvidia-smi
```

The PyTorch command below uses **CUDA 12.8**. For this setup, use a driver **570.26 or newer** as a conservative starting point for compilation support. Older drivers can have CUDA compatibility limitations, especially for newly compiled GPU code. If the driver is missing or incompatible, have the machine provider/administrator prepare a compatible driver or image before continuing.

References: [CUDA 12.8 driver information](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html), [CUDA compatibility limitations](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

### 2. Install system tools

On Ubuntu, if these tools are not already provided:

```bash
sudo apt update

sudo apt install -y \
  git \
  python3 \
  python3-dev \
  python3-venv \
  python3-pip \
  build-essential

python3 --version
```

`build-essential` supplies GCC, G++, and related build tools. These commands install development tools; they do not install an NVIDIA GPU driver.

### 3. Clone the repository

```bash
git clone https://github.com/snadiger-oss/efficientfuse.git
cd efficientfuse
```

If the repository is already cloned, enter it and update it instead:

```bash
cd efficientfuse
git pull --ff-only
```

Run all remaining commands from this repository directory.

### 4. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

### 5. Install CUDA-enabled PyTorch

Start with **PyTorch 2.10.0**, the version used for the project's local CPU checks. This is an official CUDA 12.8 wheel; its GPU behavior still needs verification on the allocated hardware.

```bash
python -m pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Source: [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/).

TorchInductor is included with PyTorch. Keep the matching Triton dependency installed by the PyTorch package. The project does not require torchvision or torchaudio.

### 6. Install GreenFuse and its dependencies

```bash
python -m pip install -e ".[config,measurement,analysis,test]"
python -m pip check
greenfuse --help
```

This installs the project in editable mode, plus:

| Dependency group | Purpose |
| --- | --- |
| `config` | PyYAML for the experiment configuration files |
| `measurement` | `nvidia-ml-py`, imported as `pynvml`, for NVIDIA energy readings |
| `analysis` | Matplotlib for result plots |
| `test` | Pytest; the provided test commands use standard-library unittest |

### 7. Verify GPU access from Python

```bash
python - <<'PY'
import torch
import triton

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("Triton:", triton.__version__)

assert torch.version.cuda is not None, "Installed PyTorch has no NVIDIA CUDA support"
assert torch.cuda.is_available(), "PyTorch cannot access the NVIDIA GPU"

x = torch.randn(1024, device="cuda:0")
y = torch.relu(x)
torch.cuda.synchronize()

print("GPU:", torch.cuda.get_device_name(0))
print("GPU execution succeeded.")
PY
```

### 8. Record the software environment

```bash
mkdir -p results/setup

python -m pip freeze > results/setup/packages.txt
git rev-parse HEAD > results/setup/commit.txt
nvidia-smi > results/setup/nvidia-smi.txt
```

### 9. Check the energy measurement environment

Reserve the **whole physical GPU** for the experiment. Pause other GPU jobs on that device while taking measurements.

`--exclusive-gpu` records your assertion that this allocation is exclusive; it does not reserve the GPU or stop other processes. The initial energy implementation does not support attribution to MIG slices or virtualized GPUs.

```bash
python scripts/check_environment.py \
  --device cuda:0 \
  --exclusive-gpu \
  --output results/setup/environment.json
```

Inspect the report. Before energy experiments, expect:

- `status` is `cuda_available`.
- `energy.counter_available` is `true`.
- `energy.reason` is `null`.
- CUDA and NVML identify the same physical GPU.

A successful command exit alone does not establish that energy readings are usable; inspect the energy fields too. Unsupported counters or a nonempty energy reason must be resolved before making energy comparisons.

### 10. Validate the execution variants and run tests

```bash
python scripts/check_execution.py \
  --device cuda:0 \
  --compiler-artifacts \
  --output results/setup/execution_checks.json
```

All eight variants should report `ready`. A `correct_profile_unverified` result means numerical checks passed but the execution mode could not be verified from profiler evidence. Inspect any failure or unverified result before benchmarking.

Then run the ordinary test suite with GPU tests enabled:

```bash
GREENFUSE_RUN_GPU_TESTS=1 \
GREENFUSE_EXCLUSIVE_GPU=1 \
  python -m unittest discover -s tests -v
```

Review failures and skipped-test reasons. The separate long planner test is disabled by this command; the next section runs the real planner and preserves its evidence.

## Run the first complete experiment

Start with [configs/smoke.yaml](configs/smoke.yaml): W1 at `[32, 256]`, five rounds, and measurement windows of at least three seconds.

### 1. Inspect the graph and requested candidates

```bash
greenfuse inspect --config configs/smoke.yaml
```

### 2. Measure, select, validate, and cache

```bash
greenfuse calibrate \
  --config configs/smoke.yaml \
  --exclusive-gpu \
  --output-dir results/w1 \
  --cache-dir results/decisions
```

Calibration measures the eight variants and the eager baseline with shuffled order in each round. Timing and energy use separate passes. Compilation, correctness checks, profiling, warm-up, and preparation trials are outside the reported performance windows. Required input copies for CUDA Graph replay remain inside every measured execution.

The original and changed input sets are used for correctness. Benchmark windows use the same resident input values and shapes across candidates.

If the first comparison is complete, calibration repeats the full set with a different order seed. The output identifies the saved `initial_report`, `validation_report`, and `decision_path`.

| Outcome | Meaning |
| --- | --- |
| `validated` and `cache_written: true` | The same median-energy winner met the runtime bound in the fresh comparison; a decision was saved. |
| `inconclusive` | Validation did not establish a stable choice, for example because the winner changed. No new decision was saved. |
| `incomplete` | Required measurements or execution evidence are missing or invalid. |
| `runtime_only` | CPU development evidence is available; no GPU energy choice can be made. |
| `invalid` | The records do not satisfy the selection contract. |

Default policy requires complete measurements for every requested compiled candidate. An optional `preparation_failures` exclusion policy must be configured before measurement. Missing energy or incomplete timing never becomes an automatic exclusion.

### 3. Run the validated cached choice

After calibration reports `validated`, use the same configuration and hardware/software environment:

```bash
greenfuse run \
  --config configs/smoke.yaml \
  --exclusive-gpu \
  --cache-dir results/decisions \
  --output-dir results/w1_cached
```

A saved decision skips the exhaustive search. A new Python process may still need to compile/capture the chosen variant. The command reports that reconstruction separately and then demonstrates reuse of the live executor in the same process. Its warm timing is diagnostic; it does not repeat energy validation.

A changed graph, input specialization, GPU identity, software version, implementation, or policy can cause a cache miss. Recalibrate with matching settings. `--calibrate-on-miss` can explicitly authorize a new search from the `run` command.

### 4. Inspect saved results and make a plot

Copy the `validation_report` path printed by calibration into `REPORT_JSON`. If no validation run occurred, use the `initial_report` path to inspect the available evidence.

```bash
REPORT_JSON="PASTE_THE_REPORT_JSON_PATH_HERE"

greenfuse report "$REPORT_JSON"

python scripts/analyze_results.py "$REPORT_JSON" \
  --plot results/w1_comparison.png
```

The reports retain raw trials, failure reasons, counts, units, setup costs, and environment details. Summaries show medians and variation. Missing energy stays missing; it is never replaced with zero. Energy is whole GPU-device energy, not whole-computer energy. A stable median choice does not by itself prove statistical significance or net savings after setup costs.

Generated measurements, compiler artifacts, plots, and decision-cache entries are ignored by Git. Save or share the `results/` evidence separately when needed.

## After the first shape works

[configs/midpoint.yaml](configs/midpoint.yaml) expands W1 to six shapes: `M = [32, 1024]` and `N = [256, 1024, 4096]`.

```bash
greenfuse calibrate \
  --config configs/midpoint.yaml \
  --exclusive-gpu \
  --output-dir results/midpoint \
  --cache-dir results/decisions
```

Each shape receives its own measurements, validation, and decision identity. This takes longer than the single-shape experiment.

## Component commands

These commands help debug individual parts. They are not additional prerequisites for a successful full calibration.

| Command | Purpose |
| --- | --- |
| `python scripts/inspect_partitions.py` | Capture and rebuild partitions, with CPU correctness checks |
| `python scripts/check_execution.py --device cuda:0` | Compile and verify the eight execution variants |
| `python scripts/measure_workload.py --device cuda:0 --exclusive-gpu` | Person 3's standalone time/energy measurement |
| `greenfuse benchmark --config configs/smoke.yaml --exclusive-gpu` | Save measurements and a provisional selection without validation/caching |
| `greenfuse select REPORT.json` | Select from saved evidence without allocating a GPU |
| `greenfuse report REPORT.json` | Produce a CSV summary from saved evidence |

`greenfuse calibrate` already performs measurement; do not run the standalone measurement script first unless debugging it independently.

To run the separate long GPU acceptance test on an exclusively allocated GPU:

```bash
GREENFUSE_RUN_GPU_TESTS=1 \
GREENFUSE_RUN_PLANNER_TESTS=1 \
GREENFUSE_EXCLUSIVE_GPU=1 \
  python -m unittest tests.gpu.test_end_to_end -v
```

This performs another full calibration/validation and can take several minutes. Inconclusive experiments are reported as unvalidated, not as successful energy demonstrations.

## CPU development

CPU development works without NVIDIA hardware. On a separate CPU environment, install the appropriate CPU/macOS PyTorch build and the project:

```bash
python -m pip install -e ".[torch,config,analysis,test]"

python scripts/inspect_partitions.py
python scripts/check_execution.py --device cpu --modes launch

greenfuse benchmark \
  --device cpu \
  --rounds 2 \
  --window-seconds 0.2

python -m unittest discover -s tests -v
```

CPU results cannot validate CUDA Graph replay, GPU energy, or a GPU-selected cache decision. Short timing windows may be marked invalid if they do not meet the configured duration; retain that evidence and rerun with longer windows.

## Code map

| Owner | Question | Main files |
| --- | --- | --- |
| Person 1 | What operation partitions can we construct? | [workloads/](src/greenfuse/workloads/), [graph/](src/greenfuse/graph/), [partition/](src/greenfuse/partition/) |
| Person 2 | Can each partition run correctly in both modes? | [compiler/](src/greenfuse/compiler/), [execution/](src/greenfuse/execution/) |
| Person 3 | How much runtime and GPU energy does it use? | [measurement/](src/greenfuse/measurement/), [results.py](src/greenfuse/results.py) |
| Person 4 | Which measured choice should we validate and reuse? | [planner/](src/greenfuse/planner/), [config.py](src/greenfuse/config.py), [cli.py](src/greenfuse/cli.py) |

The planner components are reusable, but the command-line workload registry currently supports W1 only. Supported graphs are pure, straight chains of addition/ReLU with external tensor inputs. LayerNorm/W2, branching graphs, dynamic shapes, FP16, backward passes, and arbitrary-program optimization are not implemented.

## Returning to an existing checkout

```bash
cd efficientfuse
source .venv/bin/activate
git pull --ff-only
python -m pip install -e ".[config,measurement,analysis,test]"
```

Code or dependency changes may invalidate existing decisions. Keep the same configuration when comparing runs and retain the saved software/environment records.
