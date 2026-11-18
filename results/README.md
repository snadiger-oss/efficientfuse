# Results

One folder per person. Each has its own README with settings, tables, findings, and rerun steps.

| Folder | Who | What is in it | Software |
|---|---|---|---|
| [haripreeth/](haripreeth/README.md) | Haripreeth Avarur | W1 Experiments 1–2 (last_pair + full), W1 all four groupings at 4 shapes, sin/tanh fan-out screen, setup evidence | PyTorch 2.15.0+cu132 |
| [bhargav/](bhargav/README.md) | Bhargav Keralapur Srinidhi | W1–W6, exp3–exp14 (GEMM policies, LayerNorm, fan-out, softmax, decode shapes, replay with resident inputs) | PyTorch 2.10.0+cu128 |

- **Don't pool absolute numbers across the two folders.** They used different PyTorch versions.
- **Candidate names** (`compiled_full_ordinary` = `p00_launch`, and so on) are explained in [haripreeth/README.md](haripreeth/README.md#candidate-names).
- **Not committed:** the planner decision cache, `results/decisions/`.
- **History:** what was done when, and why, is in [PROJECT_VERSION_LOG.md](../PROJECT_VERSION_LOG.md).
