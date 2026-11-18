# Project Version Log

This log records every version of the project: what was there, what changed, what we found, what went wrong, why we made each decision, and what we expected. **Add a new entry at the end each time you change code or run an experiment.** A template is at the bottom.

Results and plots are in [results/README.md](results/README.md), an index of `results/haripreeth/` and `results/bhargav/`. Paths in v1–v6 below are the paths at that time; since v7, Hari's folders live under `results/haripreeth/`. The steps to rerun everything are in [README.md](README.md#reproduce).

## Version index

| Version | Date | Who | Commit(s) | Summary |
|---|---|---|---|---|
| [v0](#v0--baseline-implementation-shreya) | 2026-10-28 → 10-29 | Shreya Nadiger | `b033bb2`, `7fd5ebd`, `e4b39fc`, `b498a57` | Full pipeline for workload W1 (all four project roles), tests, and setup README. Not yet run on a GPU. |
| [v1](#v1--first-gpu-runs-experiments-1-and-2-hari) | 2026-10-29 | Haripreeth Avarur | `5c3407b` | First real GPU runs on an A40. Fixed two bugs that blocked GPU selection. Experiment 1 (small shape) and Experiment 2 (large shapes) for the last_pair and full groupings. |
| [v2](#v2--all-four-groupings-at-all-shapes-hari) | 2026-10-29 | Haripreeth Avarur | `e8fe82b` | All four groupings × both execution modes × four shapes, in one run. All four shapes validated. |
| [v3](#v3--experiment-3-fan-out-workload-where-partial-fusion-wins-hari) | 2026-10-30 | Haripreeth Avarur | `e8fe82b` | Experiment 3: a second workload where splitting the computation beats full fusion by up to 34% in time and 30% in energy. |
| [v4](#v4--repository-cleanup-and-publishing-hari) | 2026-11-18 | Haripreeth Avarur | `e8fe82b`, `06885e7`, `7144860` | Repo cleaned and published to GitHub, plus a short README with rerun steps and this log. |
| [v5](#v5--workloads-w2w6-and-exp3exp14-bhargav) | 2026-10-29 → 11-18 | Bhargav Keralapur Srinidhi | `340602f` … `bb96c30`, `ac5a7d8` (PR #1), `479875a` | Workloads W2–W6, fan-out support in the graph code, GEMM and replay-input policies, and 12 experiments (exp3–exp14). Partial grouping wins only when full fusion forces a slower Triton GEMM. |
| [v6](#v6--merging-bhargavs-work-into-haris-branch-hari) | 2026-11-18 | Haripreeth Avarur | `68383d1` | Brought Shreya's latest `main` (with Bhargav's v5) into Hari's fork, kept Hari's file structure and docs, and prepared a pull request back to Shreya's repo. |
| [v7](#v7--one-results-folder-per-person-hari) | 2026-11-18 | Haripreeth Avarur | this commit | Moved Hari's results into `results/haripreeth/` to match `results/bhargav/`; `results/README.md` is now an index. |

## Terms used throughout

| Term | Meaning |
|---|---|
| **W1** | The first workload: `out = relu(x + bias) + residual`. `x` and `residual` are `[M, N]`, `bias` is `[N]`. FP32, contiguous, forward only. |
| **Grouping / partition** | How the three operations (bias add → ReLU → residual add) are split into separately compiled groups. Each group is compiled on its own by TorchInductor. |
| `p00` = **full** | `[add + ReLU + add]`: one group |
| `p01` = **first_pair** | `[add + ReLU] [add]`: two groups |
| `p10` = **last_pair** | `[add] [ReLU + add]`: two groups |
| `p11` = **separate** | `[add] [ReLU] [add]`: three groups |
| **launch / ordinary** | Normal execution: each compiled group is called one after another from Python. |
| **graph / replay** | The whole partition is captured once as a CUDA Graph and replayed on every call (one GPU submission instead of one per kernel). |
| **Candidate** | One grouping + one execution mode, such as `p00_launch`. W1 has 4 × 2 = 8 candidates. |
| **Eager baseline** | The original uncompiled PyTorch code. It is measured for comparison but never selected. |
| **Selection rule** | Keep every candidate whose median runtime is ≤ 1.05 × the fastest median, then pick the one with the lowest median energy. Ties are broken by runtime, then candidate ID. |
| **Validated** | A second, fresh measurement run picked the same winner, so the decision was cached. |
| **Inconclusive** | The fresh run picked a different winner, so nothing was cached. |

---

## v0 — Baseline implementation (Shreya)

**Dates:** 2026-10-28 to 2026-10-29. **Author:** Shreya Nadiger.

### The research question

Does the way GPU operations are grouped for compilation, with or without CUDA Graph replay, change GPU energy use? Can a measurement-driven planner find the lowest-energy option that is still within 5% of the fastest?

The usual assumption is that fusing everything into one kernel is always best. The project tests that assumption by measuring every grouping instead of assuming one. The README framed partial grouping as "a hypothesis to test; an energy advantage is not guaranteed."

### Commit 1: `b033bb2`, project scaffold (2026-10-28)

- Created the folder layout: `src/greenfuse/{workloads, graph, partition, compiler, execution, measurement, planner}`, plus `tests/{unit, integration, gpu}`, `configs/`, and `scripts/slurm/`.
- Added `.gitignore`. It keeps generated results, compiler caches, Slurm output, and credentials out of git.
- The work was split into four roles. Each role owns one question:

| Role | Question | Folders |
|---|---|---|
| Person 1 | What groupings can we build? | `workloads/`, `graph/`, `partition/` |
| Person 2 | Does each grouping run correctly in both modes? | `compiler/`, `execution/` |
| Person 3 | How much time and GPU energy does it use? | `measurement/`, `results.py` |
| Person 4 | Which option should we pick, validate, and reuse? | `planner/`, `config.py`, `cli.py` |

### Commit 2: `7fd5ebd`, Persons 1–3: workloads, execution, measurement (2026-10-29, 3,770 lines)

**Person 1: building the groupings**

- `workloads/definitions.py`: W1 is written once, as plain PyTorch. Groupings are generated from it automatically, never written by hand.
- `workloads/specs.py`, `inputs.py`, `registry.py`: describe inputs (shape, dtype, seed) without importing PyTorch, and create the actual tensors only when needed. Each description has a fingerprint, so results can be tied to the exact inputs.
- `graph/capture.py`: uses PyTorch FX to trace W1 into a graph of operations. `graph/ir.py` stores that graph in a portable, PyTorch-free form with a fingerprint.
- `graph/validate.py`: refuses anything except what is supported:
  - a straight chain of add/ReLU on input tensors;
  - no branching, no in-place ReLU, no `out=` mutation, no non-finite constants.

  The design rule throughout the code is to refuse clearly rather than silently do something wrong.
- `partition/enumerate.py`: lists every contiguous split of the chain. A chain of n operations has n − 1 gaps, so it has 2^(n−1) splits. W1 has 3 operations, so 4 partitions: `p00`, `p01`, `p10`, `p11`. A `max_partitions` limit (default 256) stops the search space from blowing up on longer graphs.
- `partition/build.py`: turns each partition into separate group callables, plus a plan saying which tensors feed which group.

**Person 2: compiling and running**

- `compiler/inductor.py`: compiles **each group separately** with TorchInductor (`fullgraph=True`, `dynamic=False`). The outer code that calls the groups is never compiled, so Inductor cannot fuse across our group boundaries; this is what makes the groupings real. Inductor's own CUDA Graph feature is switched off (`triton.cudagraphs=False`), so replay only happens where we ask for it. Each compilation records the torch version, CUDA build, device, and timings.
- `compiler/inspect.py`: proves what actually ran on the GPU. It runs one call under the PyTorch profiler inside a named marker (`greenfuse.logical_invocation`), then counts the GPU kernels inside that marker. It also checks:
  - ordinary mode used kernel launches and no graph launch;
  - replay mode used a graph launch.

  The rule is "never guess kernels": if the trace is ambiguous, there is no count.
- `execution/launch.py` (`LaunchExecutor`): ordinary mode. It calls the compiled groups in order on the caller's tensors.
- `execution/graphs.py` (`GraphExecutor`): replay mode.
  - Setup: copies the inputs into owned fixed buffers, warms up on a side stream, then captures the whole partition into one `torch.cuda.CUDAGraph`.
  - Every call: **copies the caller's inputs into those buffers, then replays the graph.** This was a deliberate choice, and it matters later (see the replay-copies issue in v1).
- `execution/buffers.py`: input contracts (shape, dtype, device, stride must match), fixed buffers, and stream checks.
- `execution/prepare.py`: prepares one candidate end to end (compile → build executor → check correctness → profile) and records whether it is `ready` for GPU measurement.

**Person 3: measuring**

- `measurement/timing.py`: synchronize, run the executor N times, synchronize, divide wall-clock time by N. The pilot step (`calibrate_iterations`) picks N so each window lasts at least 1.25 × the minimum window (3 s on GPU).
- `measurement/energy.py` (`NvmlEnergyMeter`):
  - reads the GPU's cumulative energy counter (NVML, in mJ) before and after a window: energy per call = Δ mJ / 1000 / N;
  - has no fallback to power × time estimates;
  - finds the GPU by exact UUID match between PyTorch and NVML, never by guessing indexes;
  - refuses MIG slices, virtualized GPUs, and MPS;
  - checks for other processes on the GPU before and after each window.
- `measurement/runner.py` (`run_measurements`): the main measurement loop.
  - Separate **timing** and **energy** passes, R rounds each. The candidate order is shuffled every round with a seed.
  - For **every trial**, the executor is rebuilt from scratch, then checked for correctness, profiled, warmed up, and measured.
  - Every trial is saved to JSON as it finishes, so a crash keeps everything measured so far.
  - Windows shorter than the minimum are marked invalid but kept.
- `measurement/baseline.py`: runs the eager workload as an external baseline. It is excluded from selection.
- `measurement/correctness.py`: compares outputs with the eager reference (`rtol=1e-5`, `atol=1e-6`), outside the measured windows.
- `measurement/environment.py`: records Python, platform, package versions, GPU, energy counter status, and relevant environment variables.
- `results.py`: the report format. Raw trials are always kept; summaries (median, min, max) are derived from them and never replace them. Every report has a fingerprint.
- Scripts: `check_environment.py`, `check_execution.py`, `inspect_partitions.py`, `measure_workload.py`, `analyze_results.py`.

### Commit 3: `e4b39fc`, Person 4: planner, selection, validation, cache (2026-10-29, 1,538 lines)

- `config.py`: strict YAML/JSON experiment configs that reject unknown keys and bad values. `configs/smoke.yaml` is W1 at 32×256 (5 rounds, 3 s windows). `configs/midpoint.yaml` is six shapes.
- `planner/candidates.py`: combines each partition with each mode, giving 8 candidates for W1.
- `planner/select.py` (`select_report`): the selection rule above. Before selecting, it checks that the evidence is consistent:
  - every requested candidate has complete timing and energy;
  - the exclusive-GPU assertion is present;
  - the CUDA, NVML, and environment GPU IDs match;
  - each candidate has real kernel and mode evidence;
  - the compiler's torch version matches the environment's torch version (this check matters in v1).

  If any check fails, the result is `invalid` or `incomplete`. It never quietly picks something.
- `validate_repeat`: the whole measurement is repeated with a different shuffle seed. The decision is `validated` only if the same winner comes out and the second run's trials started after the first run's ended.
- `planner/calibrate.py` (`greenfuse calibrate`): capture → partitions → candidates → initial measurement → selection → fresh validation → cache if validated.
- `planner/identity.py` and `planner/cache.py`: a cached decision is keyed by a fingerprint of everything that could change the answer:
  - graph, inputs, experiment config, and candidate list;
  - stable environment: versions, GPU UUID, driver, power limit, clocks;
  - a hash of all GreenFuse source files;
  - the selection policy.

  On lookup, the cached raw evidence is re-checked and re-validated. File names are these hashes, which is why `results/decisions/` has hash-named files.
- `planner/dispatch.py`: reuses a live, already-built executor within the same process (`greenfuse run`).
- `cli.py`: `greenfuse inspect | benchmark | select | calibrate | run | report`.
- Tests for config, planner, cache, and calibration (using clearly labeled synthetic data), plus a GPU end-to-end test.

### Commit 4: `b498a57`, setup and usage README (2026-10-29)

A 363-line README (now in [docs/SETUP.md](docs/SETUP.md)). It covers installation on Ubuntu with **PyTorch 2.10.0 + CUDA 12.8**, environment checks, how to run calibration, what each outcome means, CPU development, and a code map.

### State at the end of v0

- **Tests:** 117 tests run locally on 2026-10-29: 110 passed, and 7 GPU tests were skipped because no GPU was available.
- **GPU validation:** none yet. The README said: "NVIDIA GPU execution, real energy measurements, and the full GPU calibration/cache demonstration still need hardware validation." Selection and cache logic had been tested only on synthetic data.
- **Scope:** only W1 and straight add/ReLU chains. No LayerNorm, branching graphs, FP16, dynamic shapes, or backward pass.

---

## v1 — First GPU runs: Experiments 1 and 2 (Hari)

**Date:** 2026-10-29 (runs between 21:05 and 21:53). **Author:** Haripreeth Avarur. **Commit:** `5c3407b`. The same content also exists in Shreya's repo as `e17577c`, the original version of this commit, which reached her `main` through Bhargav's PR #1.

### Goal

1. Run Shreya's pipeline on a real GPU for the first time, and prove it works end to end: compile → verify → measure → select → validate → cache.
2. Cover our half of the team's search space: the **last_pair (`p10`)** and **full (`p00`)** groupings, each in both modes. The other pair of teammates was going to run first_pair (`p01`) and separate (`p11`).

### Setup used

| Item | Value |
|---|---|
| Machine | University of Michigan Great Lakes cluster, Slurm, whole GPU reserved (setup check Slurm job 65044926) |
| GPU | NVIDIA A40, 48 GB, compute capability 8.6, driver 595.71.05 |
| Software | Python 3.11.18, **PyTorch 2.15.0+cu132**, Triton 3.9.0, nvidia-ml-py 13.615.71, gcc 13.2.0, conda env `env_scalable` |
| Evidence saved | `results/setup/`: `environment.json`, `execution_checks_last_pair_and_full.json`, `packages.txt`, `commit.txt`, `nvidia-smi.txt`, Inductor debug code and profiler traces |

Setup check result: all 4 requested variants were `ready` and passed the replay check. Observed kernels per call:
- `p00` (full): **1 kernel**, `triton_poi_fused_add_relu_0`, in both modes;
- `p10` (last_pair): **2 kernels**, `triton_poi_fused_add_0` and `triton_poi_fused_add_relu_0`, in both modes.

This confirmed that the requested groups really became separate kernels.

### Problems found and how each was fixed

#### Problem 1: TorchInductor would not compile with the system compiler

- **Symptom:** compilation failed on the GPU node.
- **Cause:** Great Lakes' default system gcc is 8.5, which is too old for the C++ code that TorchInductor generates and builds.
- **Fix (environment only, no code change):** `module load gcc/13.2.0` and `export CC=$(which gcc) CXX=$(which g++)` before running. This is now part of the rerun steps.

#### Problem 2: kernel-trace parser rejected every candidate on PyTorch 2.15 (bug)

- **Symptom:** profiling reported "Expected exactly one complete logical-invocation marker". Every candidate was therefore "not ready for GPU measurement", so nothing could be measured.
- **Cause:** `compiler/inspect.py` looks for exactly one profiler event named `greenfuse.logical_invocation`. PyTorch 2.15 records that marker **twice**: once on the CPU timeline, and a mirrored copy on the GPU timeline with category `gpu_user_annotation`. The parser found 2 markers and refused.
- **Fix:** count only the CPU-side marker; ignore events with `cat == "gpu_user_annotation"`. This is a 2-line change in `summarize_kernel_trace`.
- **Why this fix:** the CPU marker is the one that brackets the whole call, which is what the parser was designed to use. All the other safety checks are unchanged (kernels must sit inside the marker, the device must match, the launch type must match the mode).

#### Problem 3: selection rejected valid runs because of a torch version string mismatch (bug)

- **Symptom:** after measuring, selection returned `invalid` because the compiler's torch version did not match the environment's.
- **Cause:** two places recorded the version differently:
  - `compiler/inductor.py` used `torch.__version__` = **`2.15.0+cu132`**;
  - `measurement/environment.py` used package metadata (`importlib.metadata.version("torch")`) = **`2.15.0`**. Pip wheels drop the local `+cu132` tag from metadata.

  `planner/select.py` requires them to be equal, so every real run was rejected. You can still see the old value in `results/setup/environment.json` (`"torch": "2.15.0"`), which was recorded before the fix. The experiment reports recorded after the fix show `2.15.0+cu132` in both places.
- **Fix:** the environment report now uses `str(torch.__version__)`, the same source as the compiler.
- **Why this fix:** the check is right in principle, because evidence must come from one PyTorch build. Only the two sources were inconsistent. Keeping the `+cu132` tag is also more precise, since it identifies the CUDA build.
- **Side effect:** planner cache keys include the torch version and a hash of the source files, so decisions from before and after this change will never match. That is intended.

#### Problem 4: no way to run only some groupings (missing feature)

- **Need:** we were splitting the four groupings between two pairs of teammates, but the planner always ran all 4.
- **Fix:**
  - an optional `partitions:` config key, for example `partitions: [p10, p00]`. `ExperimentConfig.select_partitions()` filters the enumerated partitions and errors on unknown IDs. It is used in `planner/calibrate.py`.
  - the same subset option as a `--partitions` flag on `scripts/check_execution.py` and `scripts/measure_workload.py`.
- **Why:** each pair covers its half with the same code and policy, without editing the enumeration logic. Leaving the key out keeps the old behavior of running all partitions.

#### Problem 5: raw results are hard to read (missing tooling)

- **Need:** raw JSON uses internal IDs like `p10_graph`. These can't be renamed, because the IDs are part of the fingerprints.
- **Fix:** new `scripts/export_experiment.py`. It copies a planner run folder unchanged into `raw/` and writes:
  - `summary.csv` with readable names (`compiled_last_pair_replay`, …): initial and validation passes, runtime and energy medians with min/max, and which candidate was selected;
  - `runtime_vs_energy.png`: the fastest-candidate line, the 1.05× cutoff, and the selected candidate circled.

#### Problem 6: Experiment 2 was interrupted partway through

- **What happened:** attempt 1 (21:19–21:27) finished 1024×1024 but stopped during 4096×4096. The log ends without an error message, so the session or job ended. Logged as `calibrate_attempt1_1024_done_4096_interrupted.log`.
- **Fix:** `configs/exp2_w1_large_last_pair_full_resume.yaml` is identical except that it only has the 4096 and 8192 shapes. Attempt 2 (21:37–21:53) completed both. 1024×1024 was kept from attempt 1, because each shape is calibrated and validated independently.

### Experiment 1: small shape 32×256 (inconclusive)

- **Config:** `configs/exp1_w1_small_last_pair_full.yaml`. It was originally run as `smoke_last_pair_full.yaml`, which is why the log shows that name. 5 rounds, 3 s windows, 1.05 runtime tolerance.
- **Why:** this is Shreya's default smoke shape, and the first thing to run on any new setup.
- **What we expected:** at 32×256 (8,192 elements, 32 KB per tensor) the GPU work is tiny, so each call should be dominated by CPU-side launch overhead. Replay, with one submission per call, should beat ordinary execution. The grouping itself should matter little.

| Candidate | Initial: µs / mJ per call | Validation: µs / mJ per call |
|---|---|---|
| compiled_last_pair_ordinary | 202.4 / 14.99 | 206.9 / 15.36 |
| compiled_last_pair_replay | 97.3 / 7.37 | 97.6 / 7.51 |
| compiled_full_ordinary | 122.2 / 8.86 | 121.8 / 9.05 |
| compiled_full_replay | 96.6 / 7.48 | 97.3 / 7.41 |
| eager_baseline | 86.5 / 6.38 | 86.3 / 6.52 |

- **Result: inconclusive, so nothing was cached.**
  - The initial run picked last_pair_replay (7.37 mJ vs 7.48).
  - The validation run picked full_replay (7.41 mJ vs 7.51).
  - The two are about 1.5% apart, which is within run-to-run noise, so the planner correctly refused to cache. The reason it recorded: "Fresh measurements chose a different winner; rerun or increase measurement windows."
- **What we learned:**
  - The expectation held: replay roughly halves the per-call cost compared with ordinary mode, and the grouping barely matters.
  - Ordinary-mode cost grows with the number of groups (last_pair, with 2 groups, at 202 µs vs full, with 1 group, at 122 µs). Each group is another Python → compiled-function → kernel-launch round trip.
  - Eager (86 µs) is faster than every compiled option at this size. Compiled calls carry extra Python wrapper overhead that doesn't pay off when the GPU work is about 0.1 µs.
  - An inconclusive result is the planner working correctly, not failing.

### Experiment 2: large shapes 1024², 4096², 8192² (validated)

- **Config:** `configs/exp2_w1_large_last_pair_full.yaml`, plus the resume config.
- **Why:** Experiment 1 showed that small shapes measure overhead, not fusion. Fusion's benefit is fewer trips to GPU memory, so we needed shapes where memory traffic dominates:
  - 1024² is 4 MB per tensor, which fits in the A40's 6 MB L2 cache;
  - 4096² is 64 MB and 8192² is 256 MB, both far larger than L2.
- **What we expected:** at large shapes, runtime and energy should scale with the number of tensor reads and writes per call:
  - **full** reads `x` and `residual` and writes `out`: **3 tensor passes**;
  - **last_pair** has 2 for group 1 (read `x`, write the temp) and 3 for group 2 (read the temp and `residual`, write `out`): **5 tensor passes**.

  So full should cost about 3/5 = 0.60 of last_pair.

| Shape | Selected (validated and cached) | Why |
|---|---|---|
| 1024×1024 | `compiled_full_replay` | Still overhead-bound. full_ordinary used less energy but was about 25–27% slower than the fastest, so it failed the 5% rule. |
| 4096×4096 | `compiled_full_ordinary` | Fastest and lowest energy |
| 8192×8192 | `compiled_full_ordinary` | Fastest and lowest energy |

Fresh validation pass, µs / mJ per call:

| Candidate | 1024² | 4096² | 8192² |
|---|---|---|---|
| compiled_full_ordinary | 124.2 / 12.66 | **355.9 / 78.5** | **1392 / 317** |
| compiled_full_replay | **99.1 / 14.26** | 839.7 / 187.5 | 3309 / 740 |
| compiled_last_pair_ordinary | 209.4 / 20.48 | 592.5 / 132.6 | 2333 / 525 |
| compiled_last_pair_replay | 100.3 / 16.06 | 1078.8 / 234.1 | 4251 / 949 |
| eager_baseline | 88.1 / 12.19 | 846.7 / 191.7 | 3354 / 763 |

**Findings:**
- **The prediction matched exactly.** Full vs last_pair (ordinary) is 355.9 / 592.5 = 0.60 at 4096² and 1392 / 2333 = 0.60 at 8192². Energy shows the same ratio.
- **The kernels are memory-bound.** At 4096², full moves 3 × 64 MB = 201 MB in 355.9 µs, which is about 566 GB/s, or about 81% of the A40's peak of about 696 GB/s.
- **Full uses about 59% less energy than eager** at large shapes (78.5 vs 191.7 mJ). Eager runs 3 separate kernels, one per operation.
- **Run-to-run variation is under 1%**, so these decisions are stable. All three shapes validated.
- 1024² behaves like the small shape (overhead-bound, replay wins), because 4 MB fits in L2.

### Mistake found in the measurement design: replay timings include input copies

Replay looked about 2.4× slower than ordinary at large shapes (839.7 vs 355.9 µs). That is not a real property of CUDA Graphs.

- **Cause:** `GraphExecutor.run()` copies every input into its fixed buffers on every call, by design in v0. For W1 that adds a read and a write of `x` and of `residual`: **4 extra tensor passes per call** that ordinary mode does not do. The numbers match exactly:
  - full_replay: 3 + 4 = 7 passes; predicted 355.9 × 7/3 ≈ 830 µs; **measured 840 µs**;
  - last_pair_replay: 5 + 4 = 9 passes; predicted ≈ 1066 µs; **measured 1079 µs**.
- **Effect:** "replay is slower at large shapes" must **not** be reported as a finding. Ordinary vs ordinary, and ordinary vs eager, are unaffected.
- **Status: not fixed yet, and needs a team decision.** The project spec asks to measure replay without the copy and report the buffer-refresh cost separately. Changing this alters the measurement contract and invalidates cached decisions, so it was documented rather than changed unilaterally.

### Other notes

- **Version difference from the team guide:** the team guide pins torch 2.10.0+cu128; these runs used 2.15.0+cu132. Use the same version before pooling numbers with the other pair. Cache keys include the version, so decisions from the two versions will never match.
- Everything was committed with the results:
  - raw planner output, including per-trial kernel evidence;
  - summaries, plots, and logs.

  The planner cache (`results/decisions/`) was **not** committed. It is regenerated by `greenfuse calibrate`.

---

## v2 — All four groupings at all shapes (Hari)

**Date:** 2026-10-29 (run 22:32–23:30). **Author:** Haripreeth Avarur. **Commit:** `e8fe82b`.

### Why

1. Experiments 1–2 covered only half the groupings. Comparing our half with the other pair's half from a different session would mix GPU sessions, and possibly torch versions (2.10 vs 2.15). The selection rule assumes all candidates were measured **under identical conditions, in one shuffled run**.
2. Check whether any partial grouping (first_pair or separate) ever beats full for W1.
3. Re-test the small shape with more candidates.

### What ran

- **Config:** `configs/all_groupings_w1_all_shapes.yaml`. It has no `partitions:` key, so all four groupings ran: 8 compiled candidates plus eager, at 32×256, 1024², 4096², and 8192². 5 rounds, 3 s windows, separate timing and energy passes, with fresh validation.
- **Kernels observed per call:**
  - `p00`: 1;
  - `p01` and `p10`: 2 each;
  - `p11`: 3;
  - eager: 3.

  These are the same in both modes, so each requested grouping really became that many kernels.
- **Output:** `results/all_groupings_w1_all_shapes/`.
  - Per shape: `summary.csv`, `runtime_vs_energy.png`, and `raw/`.
  - For all shapes together: `summary_all_shapes.csv` and `energy_by_shape.png`.

### What we expected

At large shapes, cost should follow the tensor passes per call: separate : pair : full = **7 : 5 : 3**, so 2.33 : 1.67 : 1. first_pair and last_pair move the same amount of data, so they should tie. Full should win everywhere, because W1 is a pure elementwise chain: fusing only removes memory traffic and costs nothing.

### Results (all four shapes validated)

| Shape | Selected |
|---|---|
| 32×256 | `compiled_full_replay` |
| 1024×1024 | `compiled_full_replay` |
| 4096×4096 | `compiled_full_ordinary` |
| 8192×8192 | `compiled_full_ordinary` |

Fresh validation pass, µs / mJ per call:

| Candidate | 32×256 | 1024² | 4096² | 8192² |
|---|---|---|---|---|
| compiled_full_ordinary | 120.6 / 9.14 | 123.3 / 12.36 | **354.6 / 78.50** | **1393 / 317.4** |
| compiled_full_replay | **95.9 / 7.31** | **96.9 / 14.23** | 839.2 / 187.2 | 3309 / 743.0 |
| compiled_first_pair_ordinary | 205.7 / 15.11 | 207.0 / 20.55 | 590.6 / 127.1 | 2332 / 504.1 |
| compiled_first_pair_replay | 96.8 / 7.36 | 98.2 / 16.08 | 1077.6 / 236.4 | 4252 / 942.4 |
| compiled_last_pair_ordinary | 204.7 / 15.51 | 207.9 / 20.48 | 590.7 / 129.2 | 2332 / 512.8 |
| compiled_last_pair_replay | 96.8 / 7.46 | 98.1 / 16.15 | 1077.6 / 238.0 | 4250 / 950.7 |
| compiled_separate_ordinary | 285.8 / 21.59 | 286.5 / 27.68 | 827.8 / 177.6 | 3275 / 721.7 |
| compiled_separate_replay | 98.6 / 7.61 | 99.4 / 17.09 | 1311.3 / 282.4 | 5190 / 1148 |
| eager_baseline | 84.9 / 6.44 | 86.1 / 12.14 | 847.6 / 191.1 | 3354 / 764.3 |

### Findings

- **The prediction matched.** At 4096², the ordinary-mode ratio is 827.8 : 590.7 : 354.6 = **2.33 : 1.67 : 1**. Energy follows the same order. first_pair and last_pair are identical within noise (590.6 vs 590.7 µs).
- **Full fusion wins at every shape. No partial grouping beat it for W1.** This was expected for a pure elementwise chain. It meant we needed a different workload where full fusion has a real cost (v3).
- **Small shapes are overhead-bound:**
  - replay makes every grouping cost about the same (96–99 µs);
  - in ordinary mode, cost rises by about 80 µs for each extra group (121 → 205 → 286 µs);
  - eager is still the fastest at these sizes.
- **32×256 validated this time** (full_replay), whereas Experiment 1 was inconclusive. The replay candidates are within about 1–3% of each other here, so whether this shape validates comes down to noise. Treat the small-shape decision as fragile.
- The replay-copies issue from v1 is visible again: full_replay 839 µs vs full_ordinary 355 µs at 4096².

---

## v3 — Experiment 3: fan-out workload where partial fusion wins (Hari)

**Date:** 2026-10-30 (Slurm job 65052608, 00:28–00:42). **Author:** Haripreeth Avarur. **Commit:** `e8fe82b`.

### Why

v2 showed that for a pure elementwise chain, full fusion always wins, because it only removes memory traffic. To test the project's actual hypothesis, that partial grouping can save energy, we needed a workload where **fusing everything has a cost**. The earlier plan named linear + GELU as one example. We chose a **fan-out** workload instead, because it creates a clear, tunable trade-off between recomputing a value and storing it, using only elementwise and reduction ops.

### The workload

```python
y   = E_k(x)                        # k chained expensive ops: sin, tanh, sin, tanh, ...
out = y * y.sum(-1, keepdim=True)   # y feeds BOTH a row reduction and a multiply
```

- **Shapes:** 8192×8192 and 2048×32768. Rows are long (8,192 and 32,768 elements), so Inductor cannot hold a whole row on-chip and uses a **looped reduction** that walks each row twice: pass 1 computes the row sum, and pass 2 computes `out`.
- **The trade-off:**
  - **Full fusion** (one kernel) does not keep `y` between the two passes. It recomputes `E_k(x)` in both, so it does **2k** expensive operations per element and reads `x` twice.
  - **Split after y**, `[E_k] [sum + multiply]`, computes `y` once (**k** operations), writes it to memory, and then the reduction kernel reads it twice. That costs extra memory traffic.
- **What we expected:** for small k, recomputing is cheap and full fusion wins. Above some k, the wasted k extra transcendental ops per element cost more than the extra memory traffic, and the split should win.

### Method

- `scripts/exp3_screen_fanout.py` runs as a **standalone screen**, outside the planner. W1's graph validator only supports straight add/ReLU chains, so this workload can't go through the planner yet.
- Each group is compiled separately with `torch.compile(backend="inductor", fullgraph=True, dynamic=False)`, which mirrors the planner's per-group compilation.
- **Candidates:**
  - `compiled_full`: one group;
  - `compiled_split_after_y`: `[E_k] [sum + multiply]`;
  - `compiled_separate`: `[E_k] [sum] [multiply]`;
  - `eager_baseline`.
- **Settings:**
  - k ∈ {2, 4, 8, 12, 16, 24, 32}; 5 rounds of 2 s windows with shuffled order; ordinary execution only.
  - Each window measures time and NVML energy together.
  - Correctness is checked against eager: maximum error relative to max|eager| must be ≤ 1e-4. All candidates passed.
  - The selection rule is the same as the planner's (1.05× runtime cutoff, then lowest energy).
- **How this differs from the planner** (this is why it is called a screen):
  - time and energy are measured in the same window, not in separate passes;
  - windows are 2 s, not 3 s;
  - there is no replay mode;
  - there is no fresh validation run.
- **Kernel analysis** explains why each result happened:
  - `scripts/exp3_inspect_kernels.py` saves the Triton code Inductor generated for every (shape, k, candidate). For each kernel it counts sin/tanh calls, loads, stores, and passes over the row, and records the profiled GPU time.
  - `scripts/exp3_trace_report.py` writes `KERNEL_TRACE_<shape>.md`, which walks through every kernel line by line, plus Chrome/Perfetto traces.

### Results

split_after_y relative to full (time / energy; **below 1 means the split wins**):

| k | 8192×8192 | 2048×32768 |
|---|---|---|
| 2 | 1.93 / 1.55 | 1.83 / 1.53 |
| 4 | 1.97 / 1.72 | 1.44 / 1.30 |
| 8 | 1.19 / 1.19 | 1.15 / 1.15 |
| 12 | **0.88 / 0.88** | **0.86 / 0.85** |
| 16 | **0.79 / 0.83** | **0.77 / 0.77** |
| 24 | **0.68 / 0.70** | **0.66 / 0.70** |
| 32 | 1.22 / 1.23 | 1.16 / 1.15 |

At 8192×8192 with k = 24: full takes 4592 µs / 1357 mJ, while split takes 3140 µs / 951 mJ.

### Findings

- **This is the first case where a partial grouping beats full fusion.** For 12 ≤ k ≤ 24, split_after_y is 12–34% faster and uses 12–30% less energy, at both shapes. The crossover is between k = 8 and k = 12.
- **The generated code confirms the cause.** At k = 24, the full kernel (`triton_red_fused_mul_sin_sum_tanh_0`) contains 24 sin + 24 tanh = **2k**, loads `x` twice, and stores once. The split's first kernel contains 12 sin + 12 tanh = **k**.
- **At k = 32, Inductor changes strategy.** The fused kernel now has 16 sin + 16 tanh = k operations and **2 stores**. It computes `y` once, stores it to a scratch buffer in pass 1, and reads it back in pass 2. Full fusion then wins again, because it gets the split's benefit without an extra kernel. Inductor's own heuristic decides between recomputing and storing, and from about k = 12 to 24 it decides wrongly for this GPU.
- **The separate grouping** (3 kernels) never wins; it pays for the extra `y.sum` round trip.
- **Small k:** full fusion wins by a lot (up to about 2×), because recomputing a few cheap ops costs less than writing and re-reading `y`.

### Caveats and next step

This is a screen, so the results are not yet planner-validated. Before claiming the result, promote the fan-out workload into the planner:
- extend the graph validator to allow fan-out and reductions;
- then use separate timing and energy passes, 3 s windows, replay mode, and fresh validation.

---

## v4 — Repository cleanup and publishing (Hari)

**Date:** 2026-11-18. **Author:** Haripreeth Avarur. **Commits:** `e8fe82b` and this log's commit.

### What changed and why

- **Published to** `https://github.com/HaripreethAvarur/efficientfuse`, on top of Shreya's existing 4 commits. Her commits and authorship are unchanged.
- **Commit authorship:** Hari's commits are authored as Haripreeth Avarur `<Hari.avarur@gmail.com>`.
- **Local junk removed:** `__pycache__/`, `.pytest_cache/`, and `src/greenfuse.egg-info/`. All are regenerated automatically.
- **`.gitignore`:** now explicitly allows the published result folders (`setup`, `exp1`, `exp2`, `all_groupings`, `exp3`). It keeps the planner decision cache, `results/decisions/`, local only, because it is machine-specific and regenerated by `greenfuse calibrate`.
- **New short [README.md](README.md):** repo structure, the main entry point (`src/greenfuse/cli.py`, the `greenfuse` command), where results are, and minimal rerun steps.
- **Shreya's detailed guide moved to [docs/SETUP.md](docs/SETUP.md).** Its links and clone URL were updated, and its content is otherwise unchanged.
- **[results/README.md](results/README.md) updated:**
  - folder layout now includes the all-groupings and Experiment 3 folders;
  - new Experiment 3 section;
  - the out-of-date "uncommitted changes" section was removed, because those changes are now committed.
- **This log was added.**

### Known gaps at this version

- `summary_all_shapes.csv` and `energy_by_shape.png`, the cross-shape tables and plots, were assembled from the per-shape summaries. No committed script produces them yet.
- The Experiment 3 Slurm script has our class account hard-coded (`--account=cse598s004w26_class`). Anyone else must change it.

---

## v5 — Workloads W2–W6 and exp3–exp14 (Bhargav)

**Dates:** 2026-10-29 to 2026-11-18. **Author:** Bhargav Keralapur Srinidhi. **Commits:** `340602f`, `a213511`, `12817cb`, `3f75afe`, `95125ce`, `7f0b37d`, `bb96c30`, merged into Shreya's `main` as PR #1 (`ac5a7d8`), then cleaned up in `479875a`. Full details, tables, and rerun steps are in [results/bhargav/README.md](results/bhargav/README.md). This entry is a summary.

### Why

v2 showed that for W1, a pure elementwise chain, full fusion always wins. To test whether partial grouping can ever win, Bhargav added workloads where full fusion might have a real cost:

| Workload | Calculation | Why full fusion might lose |
|---|---|---|
| W2 | `gelu(x @ weight + bias)` | Fusing the epilogue into the GEMM needs a Triton GEMM instead of cuBLAS |
| W3 | `gelu(layer_norm(x + residual))` | Fusing pointwise work into a reduction changes how the work is split |
| W4 | `h = x + bias; h - h.mean(-1, keepdim=True)` | `h` feeds two operations (fan-out), so full fusion may reload or recompute it |
| W5 | `softmax(gelu(x + bias), -1)` | A multi-pass reduction may recompute the pointwise work |
| W6 | `gelu(layer_norm(x + residual) @ weight + bias)` | Five operations, so 16 groupings × 2 modes = 32 candidates |

### What changed in the code

- **New operations:** matmul, GELU, LayerNorm, mean, subtract, and softmax. W3 is built per row width, because LayerNorm needs a fixed shape.
- **Fan-out support:** an operation may reuse earlier results, and groups can pass several values onward (`GroupBindings.output_ids`). This lifts the "straight chain only" limit from v0.
- **`gemm_backend` config key:** `default`, `autotune`, `triton`, or `fuse_epilogue` (a group that fuses a matmul with other operations uses a Triton GEMM; a matmul-only group keeps cuBLAS).
- **`replay_input_refresh` config key:** `always` (the default; same as v0, copies every input on every replay) or `on_change` (copies only new or modified inputs). **This addresses the replay-copies mistake found in v1.**
- **TF32 settings** are recorded in the environment report and planner identity.
- **Profiler fix for PyTorch 2.10:** after CUDA Graphs are used, later profiler sessions drop GPU kernel records. Calibration now checks the execution mode from CPU-side launch records, and authoritative kernel counts come from fresh-process checks.
- **`--partitions`** also accepts names (`full`, `first_pair`, `last_pair`, `separate`), and grouping names now live in one module, `src/greenfuse/naming.py`.
- **New scripts:** `compare_workloads.py`, `threshold_sensitivity.py`, `screen_candidates.py`, `profile_kernels.py`, `quick_energy.py`, and `scripts/slurm/overnight_bhargav.sh`. `export_experiment.py` gained a `--label` option.
- **Software:** PyTorch 2.10.0+cu128, Triton 3.6.0, Python 3.12.1 (the version in the team guide). Hari's v1–v3 runs used 2.15.0+cu132, so absolute numbers must not be pooled across the two sets. For example, eager W1 at 32×256 measured 53 µs in Bhargav's runs and 86 µs in Hari's, on the same GPU model.

### Results

**Part 1, exp3–exp6 at 4096×4096:**
- W3 (LayerNorm) and W4 (fan-out): **full fusion validated**. Compared with the best partial grouping, it is 35% and 47% faster, respectively. The suspected costs did not appear: for W4, Inductor keeps each 4096-wide row on chip, so there is no reload or recompute.
- W2 (linear + GELU): both runs inconclusive. A cuBLAS GEMM with a separate bias/GELU kernel was 8% faster and used 11% less energy than one fused Triton GEMM. Inductor's autotuner already prefers cuBLAS by default.

**Part 2, exp7–exp14** (small and decode shapes, replay with `on_change`):
- **With the default compiler policy,** partial grouping + replay never saved energy compared with full + replay.
- **With `fuse_epilogue`,** partial + replay was 13–18% faster and used 12–18% less energy at W2 64 and 128 rows and W6 64 and 512 rows. Only W2 128×4096 has a validated planner decision; the other three were inconclusive.
- **The cause:** the fused Triton GEMM is slower than cuBLAS (for example, 264.0 vs 227.0 µs at W2 128 rows).
- **Threshold sensitivity:** recomputing the selection at 1.10× and 1.20× changed no selected candidate in the 14 exported runs.

### How this relates to Hari's fan-out screen (v3)

This is Hari's reading of the two result sets side by side:
- **Bhargav's W4** has 4096-wide rows. The reduction keeps a whole row on chip, so full fusion has no cost and wins.
- **Hari's sin/tanh screen** has rows of 8,192 and 32,768. These are too long to keep on chip, so Inductor walks each row twice and recomputes the shared value, and the split wins for 12 ≤ k ≤ 24.

Together they suggest that **fan-out makes full fusion lose only when the row no longer fits on chip *and* the shared value is expensive to recompute.**

**Naming clash:** Bhargav's "exp3" is W2 linear + GELU (`configs/exp3_w2_linear_gelu_cublas.yaml`). It is not Hari's "Experiment 3" sin/tanh screen (`results/exp3_fanout_screen/`).

---

## v6 — Merging Bhargav's work into Hari's branch (Hari)

**Date:** 2026-11-18. **Author:** Haripreeth Avarur. **Commit:** `68383d1`.

### Why

Hari's fork was 4 commits ahead of Shreya's `main` and 10 behind it. Those 10 were Bhargav's v5 work, plus the original version of Hari's Experiments 1–2 commit (`e17577c`). The goal was to bring everything into one branch without losing Hari's file structure and docs, then open a pull request back to Shreya's repo.

### How conflicts were resolved

- **Merge, not rebase.** Nothing already pushed to the fork was rewritten.
- **`scripts/check_execution.py`, `scripts/measure_workload.py`, `scripts/export_experiment.py`, `src/greenfuse/config.py`: Bhargav's versions.** They build directly on the same Experiments 1–2 code (`5c3407b` and `e17577c` have identical code), adding `--workload`, partition names, and the new config keys.
- **`results/README.md`: Hari's version**, which Bhargav had not changed. It now has a naming note and points out that `replay_input_refresh: on_change` exists.
- **`README.md`: Hari's short version,** now with the W1–W6 table, Bhargav's scripts in the structure tree, and a link to his results.
- **`docs/SETUP.md`:** replaced with Bhargav's updated long guide, which has the W1–W6 status, new config keys, and Great Lakes Python-header notes.
- **`.gitignore`:** `results/bhargav/` added to the list of published result folders.

### Note on history

- **Shreya's `main` was rewritten while this merge was in progress.** All 10 new commits got new hashes, the Claude co-author lines were removed from every commit, and 3 lines of `scripts/slurm/overnight_bhargav.sh` changed. The merge was redone on top of the rewritten history, so the fork has no co-author lines from upstream.
- **The old Experiments 1–2 commit is in both histories.** Shreya's `main` contains `e17577c`, the original version of Hari's Experiments 1–2 commit, authored as `havarur@umich.edu`. Hari's fork also has its copy `5c3407b` (`Hari.avarur@gmail.com`). The code in both is identical, so the merge has both in its history.

---

## v7 — One results folder per person (Hari)

**Date:** 2026-11-18. **Author:** Haripreeth Avarur. **Commit:** this commit.

### Why

After the v6 merge, Bhargav's results were in `results/bhargav/` with their own README, but Hari's were spread across `results/` next to it. One folder per person makes it clear who ran what, under which PyTorch version, and keeps the two sets from being pooled by accident.

### What changed

- **Folders moved** (contents unchanged, git history preserved), all into `results/haripreeth/`: `setup/`, `exp1_w1_small_shape/`, `exp2_w1_large_shapes/`, `all_groupings_w1_all_shapes/`, and `exp3_fanout_screen/`.
- **`results/README.md`** became `results/haripreeth/README.md`; its links and folder tree were updated for the new location.
- **New `results/README.md`:** a short index of both folders with each person's PyTorch version, and a warning not to pool absolute numbers across them.
- **Paths updated:**
  - default output paths in `scripts/exp3_screen_fanout.py`, `exp3_inspect_kernels.py`, `exp3_trace_report.py`, and `scripts/slurm/exp3_fanout_screen.sbatch`;
  - the comment in `scripts/compare_workloads.py`;
  - the rerun commands in `README.md`;
  - three cross-references in `results/bhargav/README.md`, which pointed at Hari's old paths;
  - `.gitignore` now publishes `results/haripreeth/` and `results/bhargav/`.
- **Left unchanged:** absolute paths inside raw JSON and log files. Those are historical records of where the files were when they were made.

---

## Open issues (as of v7)

| # | Issue | Since | Notes |
|---|---|---|---|
| 1 | Replay timings include per-call input copies (4 extra tensor passes for W1) | v1 | Partly addressed in v5 by `replay_input_refresh: on_change`. The default is still `always`, and all of Hari's runs and Bhargav's exp3–exp6 used it. The team still needs to decide which policy to report. |
| 2 | Two PyTorch versions in use (Hari 2.15.0+cu132; Bhargav and the team guide 2.10.0+cu128) | v1, v5 | Don't pool absolute numbers. Rerun on one version before the final report. |
| 3 | Small-shape decisions are fragile (candidates within ~1–3%) | v1, v2, v5 | Several small-shape runs in v5 were also inconclusive. |
| 4 | The sin/tanh fan-out screen is not planner-validated | v3 | v5 added fan-out support to the graph code, so the remaining work is adding sin, tanh, and sum operations and a workload entry, then running it through `greenfuse calibrate`. |
| 5 | No script for the cross-shape summary CSV and plot of v2 | v2 | v5's `compare_workloads.py` may be adaptable. |
| 6 | Experiment names clash ("Experiment 3" vs Bhargav's exp3) | v5 | Consider renaming the sin/tanh screen to its own label (for example, exp15). |

---

## Template for the next entry

```markdown
## vN — <short title> (<who>)

**Date:** YYYY-MM-DD. **Author:** <name>. **Commit(s):** <hash>.

### Why
<What question or problem prompted this version.>

### What we expected
<The prediction before running, with the reasoning.>

### What changed
- <Code/config/script change> — <why>

### Problems / mistakes found
#### <Problem>
- **Symptom:** …
- **Cause:** …
- **Fix:** …
- **Why this fix:** …

### Results
<Config used, table of numbers, where the files are.>

### Findings
- <What the results mean; whether the expectation held.>

### Open issues / next step
- …
```
