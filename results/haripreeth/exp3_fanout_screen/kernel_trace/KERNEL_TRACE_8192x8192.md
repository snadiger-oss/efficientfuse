# Experiment 3 kernel trace — shape 8192x8192

Every row below is copied from the Triton code TorchInductor generated for this exact run
(`kernel_analysis/<shape>_k<K>/<grouping>/group*_generated.py`). The right-hand column says what each line computes.

```python
def workload(x):                       # x: FP32 matrix
    y = x
    for i in range(K):                 # K chained expensive ops
        y = torch.sin(y) if i % 2 == 0 else torch.tanh(y)
    return y * y.sum(dim=-1, keepdim=True)
```

- **Full fusion:** one compiled group `[sin/tanh chain + row sum + multiply]`
- **Partial (split after y):** `[sin/tanh chain]` `[row sum + multiply]`
- **Separate:** `[sin/tanh chain]` `[row sum]` `[multiply]`

How to read a trace: a row is too long to keep on-chip, so a reduction kernel walks it twice:
**pass 1** builds the row sum, **pass 2** multiplies and writes `out`. `y step i/K` is the value after i sin/tanh ops.
If `LOAD x` and the sin/tanh chain appear in **both** passes, the kernel is recomputing `y`.

## Overview

| K | Full fusion | Partial (split after y) | Full: time / energy | Partial: time / energy | Lower energy |
|---|---|---|---|---|---|
| 4 | 1 kernel, 8 sin/tanh (recomputes y) | 2 kernels, 4 sin/tanh (computes y once) | 991 µs / 297 mJ | 1948 µs / 509 mJ | **Full** |
| 8 | 1 kernel, 16 sin/tanh (recomputes y) | 2 kernels, 8 sin/tanh (computes y once) | 1635 µs / 488 mJ | 1945 µs / 579 mJ | **Full** |
| 12 | 1 kernel, 24 sin/tanh (recomputes y) | 2 kernels, 12 sin/tanh (computes y once) | 2372 µs / 704 mJ | 2091 µs / 622 mJ | **Partial** |
| 16 | 1 kernel, 32 sin/tanh (recomputes y) | 2 kernels, 16 sin/tanh (computes y once) | 3078 µs / 912 mJ | 2445 µs / 756 mJ | **Partial** |
| 24 | 1 kernel, 48 sin/tanh (recomputes y) | 2 kernels, 24 sin/tanh (computes y once) | 4592 µs / 1357 mJ | 3140 µs / 951 mJ | **Partial** |
| 32 | 1 kernel, 32 sin/tanh (computes y once, stores it) | 2 kernels, 32 sin/tanh (computes y once) | 3144 µs / 934 mJ | 3849 µs / 1149 mJ | **Full** |

Time and energy: screening run, medians of 5 rounds, ordinary execution, NVML GPU-device energy. GPU µs/call in the
per-kernel tables comes from a separate profiler run and excludes launch overhead.

---

## K = 4


### Full fusion — 1 kernel launch(es), 8 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_red_fused_mul_sin_sum_tanh_0` | looped row reduction | 2 | x | out | 8 | 995.5 |

**Launch 1: `triton_red_fused_mul_sin_sum_tanh_0`** (looped row reduction; group 1: inputs x → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD x ← memory [x] |
| 4 | 1 | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/4 |
| 5 | 1 | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/4 |
| 6 | 1 | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/4 |
| 7 | 1 | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/4  (= y) |
| 8 | 1 | `tmp7 = _tmp6 + tmp5` | add y step 4/4  (= y) to running row sum |
| 9 | end of 1 | `tmp6 = tl.sum(_tmp6, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 10 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 11 | 2 | `tmp8 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD x ← memory [x] |
| 12 | 2 | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 1/4 |
| 13 | 2 | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 2/4 |
| 14 | 2 | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 3/4 |
| 15 | 2 | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 4/4  (= y) |
| 16 | 2 | `tmp13 = tmp12 * tmp6` | multiply y step 4/4  (= y) × row sum Σy → out |
| 17 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp13, r0_mask)` | STORE out = y × Σy → memory [out] |

### Partial: split after y — 2 kernel launch(es), 4 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 4 | 941.0 |
| 2 | group 2 | `triton_red_fused_mul_sum_0` | looped row reduction | 2 | y | out | 0 | 994.1 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/4 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/4 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/4 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/4  (= y) |
| 6 | - | `tl.store(out_ptr0 + (x0), tmp4, None)` | STORE y step 4/4  (= y) → memory [y] |

**Launch 2: `triton_red_fused_mul_sum_0`** (looped row reduction; group 2: inputs y → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 7 | 2 | `tmp4 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 8 | 2 | `tmp5 = tmp4 * tmp2` | multiply y × row sum Σy → out |
| 9 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp5, r0_mask)` | STORE out = y × Σy → memory [out] |

### Separate (3 groups) — 3 kernel launch(es), 4 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 4 | 940.9 |
| 2 | group 2 | `triton_red_fused_sum_0` | looped row reduction | 1 | y | s | 0 | 451.4 |
| 3 | group 3 | `triton_poi_fused_mul_0` | pointwise (1 pass over elements) | 1 | s, y | out | 0 | 936.2 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/4 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/4 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/4 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/4  (= y) |
| 6 | - | `tl.store(out_ptr0 + (x0), tmp4, None)` | STORE y step 4/4  (= y) → memory [y] |

**Launch 2: `triton_red_fused_sum_0`** (looped row reduction; group 2: inputs y → output s)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 1 | `tl.store(out_ptr0 + (x0), tmp2, None)` | STORE row sum Σy → memory [s] |

**Launch 3: `triton_poi_fused_mul_0`** (pointwise (1 pass over elements); group 3: inputs y, s → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x2), None)` | LOAD y ← memory [y] |
| 2 | - | `tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')` | LOAD s ← memory [s] |
| 3 | - | `tmp2 = tmp0 * tmp1` | multiply y × s → out |
| 4 | - | `tl.store(out_ptr0 + (x2), tmp2, None)` | STORE out = y × Σy → memory [out] |

---

## K = 8


### Full fusion — 1 kernel launch(es), 16 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_red_fused_mul_sin_sum_tanh_0` | looped row reduction | 2 | x | out | 16 | 1484.0 |

**Launch 1: `triton_red_fused_mul_sin_sum_tanh_0`** (looped row reduction; group 1: inputs x → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD x ← memory [x] |
| 4 | 1 | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/8 |
| 5 | 1 | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/8 |
| 6 | 1 | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/8 |
| 7 | 1 | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/8 |
| 8 | 1 | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/8 |
| 9 | 1 | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/8 |
| 10 | 1 | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/8 |
| 11 | 1 | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/8  (= y) |
| 12 | 1 | `tmp11 = _tmp10 + tmp9` | add y step 8/8  (= y) to running row sum |
| 13 | end of 1 | `tmp10 = tl.sum(_tmp10, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 14 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 15 | 2 | `tmp12 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD x ← memory [x] |
| 16 | 2 | `tmp13 = tl_math.sin(tmp12)` | sin  → y step 1/8 |
| 17 | 2 | `tmp14 = libdevice.tanh(tmp13)` | tanh  → y step 2/8 |
| 18 | 2 | `tmp15 = tl_math.sin(tmp14)` | sin  → y step 3/8 |
| 19 | 2 | `tmp16 = libdevice.tanh(tmp15)` | tanh  → y step 4/8 |
| 20 | 2 | `tmp17 = tl_math.sin(tmp16)` | sin  → y step 5/8 |
| 21 | 2 | `tmp18 = libdevice.tanh(tmp17)` | tanh  → y step 6/8 |
| 22 | 2 | `tmp19 = tl_math.sin(tmp18)` | sin  → y step 7/8 |
| 23 | 2 | `tmp20 = libdevice.tanh(tmp19)` | tanh  → y step 8/8  (= y) |
| 24 | 2 | `tmp21 = tmp20 * tmp10` | multiply y step 8/8  (= y) × row sum Σy → out |
| 25 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp21, r0_mask)` | STORE out = y × Σy → memory [out] |

### Partial: split after y — 2 kernel launch(es), 8 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 8 | 936.9 |
| 2 | group 2 | `triton_red_fused_mul_sum_0` | looped row reduction | 2 | y | out | 0 | 995.6 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/8 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/8 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/8 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/8 |
| 6 | - | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/8 |
| 7 | - | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/8 |
| 8 | - | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/8 |
| 9 | - | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/8  (= y) |
| 10 | - | `tl.store(out_ptr0 + (x0), tmp8, None)` | STORE y step 8/8  (= y) → memory [y] |

**Launch 2: `triton_red_fused_mul_sum_0`** (looped row reduction; group 2: inputs y → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 7 | 2 | `tmp4 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 8 | 2 | `tmp5 = tmp4 * tmp2` | multiply y × row sum Σy → out |
| 9 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp5, r0_mask)` | STORE out = y × Σy → memory [out] |

### Separate (3 groups) — 3 kernel launch(es), 8 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 8 | 937.0 |
| 2 | group 2 | `triton_red_fused_sum_0` | looped row reduction | 1 | y | s | 0 | 451.9 |
| 3 | group 3 | `triton_poi_fused_mul_0` | pointwise (1 pass over elements) | 1 | s, y | out | 0 | 933.7 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/8 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/8 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/8 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/8 |
| 6 | - | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/8 |
| 7 | - | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/8 |
| 8 | - | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/8 |
| 9 | - | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/8  (= y) |
| 10 | - | `tl.store(out_ptr0 + (x0), tmp8, None)` | STORE y step 8/8  (= y) → memory [y] |

**Launch 2: `triton_red_fused_sum_0`** (looped row reduction; group 2: inputs y → output s)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 1 | `tl.store(out_ptr0 + (x0), tmp2, None)` | STORE row sum Σy → memory [s] |

**Launch 3: `triton_poi_fused_mul_0`** (pointwise (1 pass over elements); group 3: inputs y, s → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x2), None)` | LOAD y ← memory [y] |
| 2 | - | `tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')` | LOAD s ← memory [s] |
| 3 | - | `tmp2 = tmp0 * tmp1` | multiply y × s → out |
| 4 | - | `tl.store(out_ptr0 + (x2), tmp2, None)` | STORE out = y × Σy → memory [out] |

---

## K = 12


### Full fusion — 1 kernel launch(es), 24 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_red_fused_mul_sin_sum_tanh_0` | looped row reduction | 2 | x | out | 24 | 2198.5 |

**Launch 1: `triton_red_fused_mul_sin_sum_tanh_0`** (looped row reduction; group 1: inputs x → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD x ← memory [x] |
| 4 | 1 | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/12 |
| 5 | 1 | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/12 |
| 6 | 1 | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/12 |
| 7 | 1 | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/12 |
| 8 | 1 | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/12 |
| 9 | 1 | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/12 |
| 10 | 1 | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/12 |
| 11 | 1 | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/12 |
| 12 | 1 | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/12 |
| 13 | 1 | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/12 |
| 14 | 1 | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/12 |
| 15 | 1 | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/12  (= y) |
| 16 | 1 | `tmp15 = _tmp14 + tmp13` | add y step 12/12  (= y) to running row sum |
| 17 | end of 1 | `tmp14 = tl.sum(_tmp14, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 18 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 19 | 2 | `tmp16 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD x ← memory [x] |
| 20 | 2 | `tmp17 = tl_math.sin(tmp16)` | sin  → y step 1/12 |
| 21 | 2 | `tmp18 = libdevice.tanh(tmp17)` | tanh  → y step 2/12 |
| 22 | 2 | `tmp19 = tl_math.sin(tmp18)` | sin  → y step 3/12 |
| 23 | 2 | `tmp20 = libdevice.tanh(tmp19)` | tanh  → y step 4/12 |
| 24 | 2 | `tmp21 = tl_math.sin(tmp20)` | sin  → y step 5/12 |
| 25 | 2 | `tmp22 = libdevice.tanh(tmp21)` | tanh  → y step 6/12 |
| 26 | 2 | `tmp23 = tl_math.sin(tmp22)` | sin  → y step 7/12 |
| 27 | 2 | `tmp24 = libdevice.tanh(tmp23)` | tanh  → y step 8/12 |
| 28 | 2 | `tmp25 = tl_math.sin(tmp24)` | sin  → y step 9/12 |
| 29 | 2 | `tmp26 = libdevice.tanh(tmp25)` | tanh  → y step 10/12 |
| 30 | 2 | `tmp27 = tl_math.sin(tmp26)` | sin  → y step 11/12 |
| 31 | 2 | `tmp28 = libdevice.tanh(tmp27)` | tanh  → y step 12/12  (= y) |
| 32 | 2 | `tmp29 = tmp28 * tmp14` | multiply y step 12/12  (= y) × row sum Σy → out |
| 33 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp29, r0_mask)` | STORE out = y × Σy → memory [out] |

### Partial: split after y — 2 kernel launch(es), 12 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 12 | 1098.3 |
| 2 | group 2 | `triton_red_fused_mul_sum_0` | looped row reduction | 2 | y | out | 0 | 997.9 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/12 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/12 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/12 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/12 |
| 6 | - | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/12 |
| 7 | - | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/12 |
| 8 | - | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/12 |
| 9 | - | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/12 |
| 10 | - | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/12 |
| 11 | - | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/12 |
| 12 | - | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/12 |
| 13 | - | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/12  (= y) |
| 14 | - | `tl.store(out_ptr0 + (x0), tmp12, None)` | STORE y step 12/12  (= y) → memory [y] |

**Launch 2: `triton_red_fused_mul_sum_0`** (looped row reduction; group 2: inputs y → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 7 | 2 | `tmp4 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 8 | 2 | `tmp5 = tmp4 * tmp2` | multiply y × row sum Σy → out |
| 9 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp5, r0_mask)` | STORE out = y × Σy → memory [out] |

### Separate (3 groups) — 3 kernel launch(es), 12 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 12 | 1097.8 |
| 2 | group 2 | `triton_red_fused_sum_0` | looped row reduction | 1 | y | s | 0 | 451.6 |
| 3 | group 3 | `triton_poi_fused_mul_0` | pointwise (1 pass over elements) | 1 | s, y | out | 0 | 939.4 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/12 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/12 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/12 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/12 |
| 6 | - | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/12 |
| 7 | - | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/12 |
| 8 | - | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/12 |
| 9 | - | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/12 |
| 10 | - | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/12 |
| 11 | - | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/12 |
| 12 | - | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/12 |
| 13 | - | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/12  (= y) |
| 14 | - | `tl.store(out_ptr0 + (x0), tmp12, None)` | STORE y step 12/12  (= y) → memory [y] |

**Launch 2: `triton_red_fused_sum_0`** (looped row reduction; group 2: inputs y → output s)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 1 | `tl.store(out_ptr0 + (x0), tmp2, None)` | STORE row sum Σy → memory [s] |

**Launch 3: `triton_poi_fused_mul_0`** (pointwise (1 pass over elements); group 3: inputs y, s → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x2), None)` | LOAD y ← memory [y] |
| 2 | - | `tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')` | LOAD s ← memory [s] |
| 3 | - | `tmp2 = tmp0 * tmp1` | multiply y × s → out |
| 4 | - | `tl.store(out_ptr0 + (x2), tmp2, None)` | STORE out = y × Σy → memory [out] |

---

## K = 16


### Full fusion — 1 kernel launch(es), 32 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_red_fused_mul_sin_sum_tanh_0` | looped row reduction | 2 | x | out | 32 | 2925.6 |

**Launch 1: `triton_red_fused_mul_sin_sum_tanh_0`** (looped row reduction; group 1: inputs x → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD x ← memory [x] |
| 4 | 1 | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/16 |
| 5 | 1 | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/16 |
| 6 | 1 | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/16 |
| 7 | 1 | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/16 |
| 8 | 1 | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/16 |
| 9 | 1 | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/16 |
| 10 | 1 | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/16 |
| 11 | 1 | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/16 |
| 12 | 1 | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/16 |
| 13 | 1 | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/16 |
| 14 | 1 | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/16 |
| 15 | 1 | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/16 |
| 16 | 1 | `tmp13 = tl_math.sin(tmp12)` | sin  → y step 13/16 |
| 17 | 1 | `tmp14 = libdevice.tanh(tmp13)` | tanh  → y step 14/16 |
| 18 | 1 | `tmp15 = tl_math.sin(tmp14)` | sin  → y step 15/16 |
| 19 | 1 | `tmp16 = libdevice.tanh(tmp15)` | tanh  → y step 16/16  (= y) |
| 20 | 1 | `tmp19 = _tmp18 + tmp17` | add y step 16/16  (= y) to running row sum |
| 21 | end of 1 | `tmp18 = tl.sum(_tmp18, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 22 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 23 | 2 | `tmp20 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD x ← memory [x] |
| 24 | 2 | `tmp21 = tl_math.sin(tmp20)` | sin  → y step 1/16 |
| 25 | 2 | `tmp22 = libdevice.tanh(tmp21)` | tanh  → y step 2/16 |
| 26 | 2 | `tmp23 = tl_math.sin(tmp22)` | sin  → y step 3/16 |
| 27 | 2 | `tmp24 = libdevice.tanh(tmp23)` | tanh  → y step 4/16 |
| 28 | 2 | `tmp25 = tl_math.sin(tmp24)` | sin  → y step 5/16 |
| 29 | 2 | `tmp26 = libdevice.tanh(tmp25)` | tanh  → y step 6/16 |
| 30 | 2 | `tmp27 = tl_math.sin(tmp26)` | sin  → y step 7/16 |
| 31 | 2 | `tmp28 = libdevice.tanh(tmp27)` | tanh  → y step 8/16 |
| 32 | 2 | `tmp29 = tl_math.sin(tmp28)` | sin  → y step 9/16 |
| 33 | 2 | `tmp30 = libdevice.tanh(tmp29)` | tanh  → y step 10/16 |
| 34 | 2 | `tmp31 = tl_math.sin(tmp30)` | sin  → y step 11/16 |
| 35 | 2 | `tmp32 = libdevice.tanh(tmp31)` | tanh  → y step 12/16 |
| 36 | 2 | `tmp33 = tl_math.sin(tmp32)` | sin  → y step 13/16 |
| 37 | 2 | `tmp34 = libdevice.tanh(tmp33)` | tanh  → y step 14/16 |
| 38 | 2 | `tmp35 = tl_math.sin(tmp34)` | sin  → y step 15/16 |
| 39 | 2 | `tmp36 = libdevice.tanh(tmp35)` | tanh  → y step 16/16  (= y) |
| 40 | 2 | `tmp37 = tmp36 * tmp18` | multiply y step 16/16  (= y) × row sum Σy → out |
| 41 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp37, r0_mask)` | STORE out = y × Σy → memory [out] |

### Partial: split after y — 2 kernel launch(es), 16 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 16 | 1437.9 |
| 2 | group 2 | `triton_red_fused_mul_sum_0` | looped row reduction | 2 | y | out | 0 | 997.9 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/16 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/16 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/16 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/16 |
| 6 | - | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/16 |
| 7 | - | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/16 |
| 8 | - | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/16 |
| 9 | - | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/16 |
| 10 | - | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/16 |
| 11 | - | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/16 |
| 12 | - | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/16 |
| 13 | - | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/16 |
| 14 | - | `tmp13 = tl_math.sin(tmp12)` | sin  → y step 13/16 |
| 15 | - | `tmp14 = libdevice.tanh(tmp13)` | tanh  → y step 14/16 |
| 16 | - | `tmp15 = tl_math.sin(tmp14)` | sin  → y step 15/16 |
| 17 | - | `tmp16 = libdevice.tanh(tmp15)` | tanh  → y step 16/16  (= y) |
| 18 | - | `tl.store(out_ptr0 + (x0), tmp16, None)` | STORE y step 16/16  (= y) → memory [y] |

**Launch 2: `triton_red_fused_mul_sum_0`** (looped row reduction; group 2: inputs y → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 7 | 2 | `tmp4 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 8 | 2 | `tmp5 = tmp4 * tmp2` | multiply y × row sum Σy → out |
| 9 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp5, r0_mask)` | STORE out = y × Σy → memory [out] |

### Separate (3 groups) — 3 kernel launch(es), 16 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 16 | 1436.7 |
| 2 | group 2 | `triton_red_fused_sum_0` | looped row reduction | 1 | y | s | 0 | 451.2 |
| 3 | group 3 | `triton_poi_fused_mul_0` | pointwise (1 pass over elements) | 1 | s, y | out | 0 | 934.0 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/16 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/16 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/16 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/16 |
| 6 | - | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/16 |
| 7 | - | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/16 |
| 8 | - | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/16 |
| 9 | - | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/16 |
| 10 | - | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/16 |
| 11 | - | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/16 |
| 12 | - | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/16 |
| 13 | - | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/16 |
| 14 | - | `tmp13 = tl_math.sin(tmp12)` | sin  → y step 13/16 |
| 15 | - | `tmp14 = libdevice.tanh(tmp13)` | tanh  → y step 14/16 |
| 16 | - | `tmp15 = tl_math.sin(tmp14)` | sin  → y step 15/16 |
| 17 | - | `tmp16 = libdevice.tanh(tmp15)` | tanh  → y step 16/16  (= y) |
| 18 | - | `tl.store(out_ptr0 + (x0), tmp16, None)` | STORE y step 16/16  (= y) → memory [y] |

**Launch 2: `triton_red_fused_sum_0`** (looped row reduction; group 2: inputs y → output s)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 1 | `tl.store(out_ptr0 + (x0), tmp2, None)` | STORE row sum Σy → memory [s] |

**Launch 3: `triton_poi_fused_mul_0`** (pointwise (1 pass over elements); group 3: inputs y, s → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x2), None)` | LOAD y ← memory [y] |
| 2 | - | `tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')` | LOAD s ← memory [s] |
| 3 | - | `tmp2 = tmp0 * tmp1` | multiply y × s → out |
| 4 | - | `tl.store(out_ptr0 + (x2), tmp2, None)` | STORE out = y × Σy → memory [out] |

---

## K = 24


### Full fusion — 1 kernel launch(es), 48 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_red_fused_mul_sin_sum_tanh_0` | looped row reduction | 2 | x | out | 48 | 4469.6 |

**Launch 1: `triton_red_fused_mul_sin_sum_tanh_0`** (looped row reduction; group 1: inputs x → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD x ← memory [x] |
| 4 | 1 | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/24 |
| 5 | 1 | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/24 |
| 6 | 1 | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/24 |
| 7 | 1 | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/24 |
| 8 | 1 | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/24 |
| 9 | 1 | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/24 |
| 10 | 1 | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/24 |
| 11 | 1 | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/24 |
| 12 | 1 | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/24 |
| 13 | 1 | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/24 |
| 14 | 1 | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/24 |
| 15 | 1 | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/24 |
| 16 | 1 | `tmp13 = tl_math.sin(tmp12)` | sin  → y step 13/24 |
| 17 | 1 | `tmp14 = libdevice.tanh(tmp13)` | tanh  → y step 14/24 |
| 18 | 1 | `tmp15 = tl_math.sin(tmp14)` | sin  → y step 15/24 |
| 19 | 1 | `tmp16 = libdevice.tanh(tmp15)` | tanh  → y step 16/24 |
| 20 | 1 | `tmp17 = tl_math.sin(tmp16)` | sin  → y step 17/24 |
| 21 | 1 | `tmp18 = libdevice.tanh(tmp17)` | tanh  → y step 18/24 |
| 22 | 1 | `tmp19 = tl_math.sin(tmp18)` | sin  → y step 19/24 |
| 23 | 1 | `tmp20 = libdevice.tanh(tmp19)` | tanh  → y step 20/24 |
| 24 | 1 | `tmp21 = tl_math.sin(tmp20)` | sin  → y step 21/24 |
| 25 | 1 | `tmp22 = libdevice.tanh(tmp21)` | tanh  → y step 22/24 |
| 26 | 1 | `tmp23 = tl_math.sin(tmp22)` | sin  → y step 23/24 |
| 27 | 1 | `tmp24 = libdevice.tanh(tmp23)` | tanh  → y step 24/24  (= y) |
| 28 | 1 | `tmp27 = _tmp26 + tmp25` | add y step 24/24  (= y) to running row sum |
| 29 | end of 1 | `tmp26 = tl.sum(_tmp26, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 30 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 31 | 2 | `tmp28 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD x ← memory [x] |
| 32 | 2 | `tmp29 = tl_math.sin(tmp28)` | sin  → y step 1/24 |
| 33 | 2 | `tmp30 = libdevice.tanh(tmp29)` | tanh  → y step 2/24 |
| 34 | 2 | `tmp31 = tl_math.sin(tmp30)` | sin  → y step 3/24 |
| 35 | 2 | `tmp32 = libdevice.tanh(tmp31)` | tanh  → y step 4/24 |
| 36 | 2 | `tmp33 = tl_math.sin(tmp32)` | sin  → y step 5/24 |
| 37 | 2 | `tmp34 = libdevice.tanh(tmp33)` | tanh  → y step 6/24 |
| 38 | 2 | `tmp35 = tl_math.sin(tmp34)` | sin  → y step 7/24 |
| 39 | 2 | `tmp36 = libdevice.tanh(tmp35)` | tanh  → y step 8/24 |
| 40 | 2 | `tmp37 = tl_math.sin(tmp36)` | sin  → y step 9/24 |
| 41 | 2 | `tmp38 = libdevice.tanh(tmp37)` | tanh  → y step 10/24 |
| 42 | 2 | `tmp39 = tl_math.sin(tmp38)` | sin  → y step 11/24 |
| 43 | 2 | `tmp40 = libdevice.tanh(tmp39)` | tanh  → y step 12/24 |
| 44 | 2 | `tmp41 = tl_math.sin(tmp40)` | sin  → y step 13/24 |
| 45 | 2 | `tmp42 = libdevice.tanh(tmp41)` | tanh  → y step 14/24 |
| 46 | 2 | `tmp43 = tl_math.sin(tmp42)` | sin  → y step 15/24 |
| 47 | 2 | `tmp44 = libdevice.tanh(tmp43)` | tanh  → y step 16/24 |
| 48 | 2 | `tmp45 = tl_math.sin(tmp44)` | sin  → y step 17/24 |
| 49 | 2 | `tmp46 = libdevice.tanh(tmp45)` | tanh  → y step 18/24 |
| 50 | 2 | `tmp47 = tl_math.sin(tmp46)` | sin  → y step 19/24 |
| 51 | 2 | `tmp48 = libdevice.tanh(tmp47)` | tanh  → y step 20/24 |
| 52 | 2 | `tmp49 = tl_math.sin(tmp48)` | sin  → y step 21/24 |
| 53 | 2 | `tmp50 = libdevice.tanh(tmp49)` | tanh  → y step 22/24 |
| 54 | 2 | `tmp51 = tl_math.sin(tmp50)` | sin  → y step 23/24 |
| 55 | 2 | `tmp52 = libdevice.tanh(tmp51)` | tanh  → y step 24/24  (= y) |
| 56 | 2 | `tmp53 = tmp52 * tmp26` | multiply y step 24/24  (= y) × row sum Σy → out |
| 57 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp53, r0_mask)` | STORE out = y × Σy → memory [out] |

### Partial: split after y — 2 kernel launch(es), 24 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 24 | 2116.7 |
| 2 | group 2 | `triton_red_fused_mul_sum_0` | looped row reduction | 2 | y | out | 0 | 997.3 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/24 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/24 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/24 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/24 |
| 6 | - | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/24 |
| 7 | - | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/24 |
| 8 | - | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/24 |
| 9 | - | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/24 |
| 10 | - | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/24 |
| 11 | - | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/24 |
| 12 | - | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/24 |
| 13 | - | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/24 |
| 14 | - | `tmp13 = tl_math.sin(tmp12)` | sin  → y step 13/24 |
| 15 | - | `tmp14 = libdevice.tanh(tmp13)` | tanh  → y step 14/24 |
| 16 | - | `tmp15 = tl_math.sin(tmp14)` | sin  → y step 15/24 |
| 17 | - | `tmp16 = libdevice.tanh(tmp15)` | tanh  → y step 16/24 |
| 18 | - | `tmp17 = tl_math.sin(tmp16)` | sin  → y step 17/24 |
| 19 | - | `tmp18 = libdevice.tanh(tmp17)` | tanh  → y step 18/24 |
| 20 | - | `tmp19 = tl_math.sin(tmp18)` | sin  → y step 19/24 |
| 21 | - | `tmp20 = libdevice.tanh(tmp19)` | tanh  → y step 20/24 |
| 22 | - | `tmp21 = tl_math.sin(tmp20)` | sin  → y step 21/24 |
| 23 | - | `tmp22 = libdevice.tanh(tmp21)` | tanh  → y step 22/24 |
| 24 | - | `tmp23 = tl_math.sin(tmp22)` | sin  → y step 23/24 |
| 25 | - | `tmp24 = libdevice.tanh(tmp23)` | tanh  → y step 24/24  (= y) |
| 26 | - | `tl.store(out_ptr0 + (x0), tmp24, None)` | STORE y step 24/24  (= y) → memory [y] |

**Launch 2: `triton_red_fused_mul_sum_0`** (looped row reduction; group 2: inputs y → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 7 | 2 | `tmp4 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 8 | 2 | `tmp5 = tmp4 * tmp2` | multiply y × row sum Σy → out |
| 9 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp5, r0_mask)` | STORE out = y × Σy → memory [out] |

### Separate (3 groups) — 3 kernel launch(es), 24 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 24 | 2117.1 |
| 2 | group 2 | `triton_red_fused_sum_0` | looped row reduction | 1 | y | s | 0 | 451.5 |
| 3 | group 3 | `triton_poi_fused_mul_0` | pointwise (1 pass over elements) | 1 | s, y | out | 0 | 939.7 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/24 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/24 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/24 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/24 |
| 6 | - | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/24 |
| 7 | - | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/24 |
| 8 | - | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/24 |
| 9 | - | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/24 |
| 10 | - | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/24 |
| 11 | - | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/24 |
| 12 | - | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/24 |
| 13 | - | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/24 |
| 14 | - | `tmp13 = tl_math.sin(tmp12)` | sin  → y step 13/24 |
| 15 | - | `tmp14 = libdevice.tanh(tmp13)` | tanh  → y step 14/24 |
| 16 | - | `tmp15 = tl_math.sin(tmp14)` | sin  → y step 15/24 |
| 17 | - | `tmp16 = libdevice.tanh(tmp15)` | tanh  → y step 16/24 |
| 18 | - | `tmp17 = tl_math.sin(tmp16)` | sin  → y step 17/24 |
| 19 | - | `tmp18 = libdevice.tanh(tmp17)` | tanh  → y step 18/24 |
| 20 | - | `tmp19 = tl_math.sin(tmp18)` | sin  → y step 19/24 |
| 21 | - | `tmp20 = libdevice.tanh(tmp19)` | tanh  → y step 20/24 |
| 22 | - | `tmp21 = tl_math.sin(tmp20)` | sin  → y step 21/24 |
| 23 | - | `tmp22 = libdevice.tanh(tmp21)` | tanh  → y step 22/24 |
| 24 | - | `tmp23 = tl_math.sin(tmp22)` | sin  → y step 23/24 |
| 25 | - | `tmp24 = libdevice.tanh(tmp23)` | tanh  → y step 24/24  (= y) |
| 26 | - | `tl.store(out_ptr0 + (x0), tmp24, None)` | STORE y step 24/24  (= y) → memory [y] |

**Launch 2: `triton_red_fused_sum_0`** (looped row reduction; group 2: inputs y → output s)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 1 | `tl.store(out_ptr0 + (x0), tmp2, None)` | STORE row sum Σy → memory [s] |

**Launch 3: `triton_poi_fused_mul_0`** (pointwise (1 pass over elements); group 3: inputs y, s → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x2), None)` | LOAD y ← memory [y] |
| 2 | - | `tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')` | LOAD s ← memory [s] |
| 3 | - | `tmp2 = tmp0 * tmp1` | multiply y × s → out |
| 4 | - | `tl.store(out_ptr0 + (x2), tmp2, None)` | STORE out = y × Σy → memory [out] |

---

## K = 32


### Full fusion — 1 kernel launch(es), 32 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_red_fused_mul_sin_sum_tanh_0` | looped row reduction | 2 | out buffer, reused to hold y, x | out buffer, reused to hold y | 32 | 2972.2 |

**Launch 1: `triton_red_fused_mul_sin_sum_tanh_0`** (looped row reduction; group 1: inputs x → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp34 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD x ← memory [x] |
| 4 | 1 | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/32 |
| 5 | 1 | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/32 |
| 6 | 1 | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/32 |
| 7 | 1 | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/32 |
| 8 | 1 | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/32 |
| 9 | 1 | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/32 |
| 10 | 1 | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/32 |
| 11 | 1 | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/32 |
| 12 | 1 | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/32 |
| 13 | 1 | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/32 |
| 14 | 1 | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/32 |
| 15 | 1 | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/32 |
| 16 | 1 | `tmp13 = tl_math.sin(tmp12)` | sin  → y step 13/32 |
| 17 | 1 | `tmp14 = libdevice.tanh(tmp13)` | tanh  → y step 14/32 |
| 18 | 1 | `tmp15 = tl_math.sin(tmp14)` | sin  → y step 15/32 |
| 19 | 1 | `tmp16 = libdevice.tanh(tmp15)` | tanh  → y step 16/32 |
| 20 | 1 | `tmp17 = tl_math.sin(tmp16)` | sin  → y step 17/32 |
| 21 | 1 | `tmp18 = libdevice.tanh(tmp17)` | tanh  → y step 18/32 |
| 22 | 1 | `tmp19 = tl_math.sin(tmp18)` | sin  → y step 19/32 |
| 23 | 1 | `tmp20 = libdevice.tanh(tmp19)` | tanh  → y step 20/32 |
| 24 | 1 | `tmp21 = tl_math.sin(tmp20)` | sin  → y step 21/32 |
| 25 | 1 | `tmp22 = libdevice.tanh(tmp21)` | tanh  → y step 22/32 |
| 26 | 1 | `tmp23 = tl_math.sin(tmp22)` | sin  → y step 23/32 |
| 27 | 1 | `tmp24 = libdevice.tanh(tmp23)` | tanh  → y step 24/32 |
| 28 | 1 | `tmp25 = tl_math.sin(tmp24)` | sin  → y step 25/32 |
| 29 | 1 | `tmp26 = libdevice.tanh(tmp25)` | tanh  → y step 26/32 |
| 30 | 1 | `tmp27 = tl_math.sin(tmp26)` | sin  → y step 27/32 |
| 31 | 1 | `tmp28 = libdevice.tanh(tmp27)` | tanh  → y step 28/32 |
| 32 | 1 | `tmp29 = tl_math.sin(tmp28)` | sin  → y step 29/32 |
| 33 | 1 | `tmp30 = libdevice.tanh(tmp29)` | tanh  → y step 30/32 |
| 34 | 1 | `tmp31 = tl_math.sin(tmp30)` | sin  → y step 31/32 |
| 35 | 1 | `tmp32 = libdevice.tanh(tmp31)` | tanh  → y step 32/32  (= y) |
| 36 | 1 | `tmp35 = _tmp34 + tmp33` | add y step 32/32  (= y) to running row sum |
| 37 | 1 | `tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp32, r0_mask)` | STORE y step 32/32  (= y) → memory [out buffer, reused to hold y] |
| 38 | end of 1 | `tmp34 = tl.sum(_tmp34, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 39 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 40 | 2 | `tmp36 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y step 32/32  (= y) ← memory [out buffer, reused to hold y] |
| 41 | 2 | `tmp37 = tmp36 * tmp34` | multiply y step 32/32  (= y) × row sum Σy → out |
| 42 | 2 | `tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp37, r0_mask)` | STORE out = y × Σy → memory [out buffer, reused to hold y] |

### Partial: split after y — 2 kernel launch(es), 32 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 32 | 2809.3 |
| 2 | group 2 | `triton_red_fused_mul_sum_0` | looped row reduction | 2 | y | out | 0 | 997.1 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/32 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/32 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/32 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/32 |
| 6 | - | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/32 |
| 7 | - | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/32 |
| 8 | - | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/32 |
| 9 | - | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/32 |
| 10 | - | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/32 |
| 11 | - | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/32 |
| 12 | - | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/32 |
| 13 | - | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/32 |
| 14 | - | `tmp13 = tl_math.sin(tmp12)` | sin  → y step 13/32 |
| 15 | - | `tmp14 = libdevice.tanh(tmp13)` | tanh  → y step 14/32 |
| 16 | - | `tmp15 = tl_math.sin(tmp14)` | sin  → y step 15/32 |
| 17 | - | `tmp16 = libdevice.tanh(tmp15)` | tanh  → y step 16/32 |
| 18 | - | `tmp17 = tl_math.sin(tmp16)` | sin  → y step 17/32 |
| 19 | - | `tmp18 = libdevice.tanh(tmp17)` | tanh  → y step 18/32 |
| 20 | - | `tmp19 = tl_math.sin(tmp18)` | sin  → y step 19/32 |
| 21 | - | `tmp20 = libdevice.tanh(tmp19)` | tanh  → y step 20/32 |
| 22 | - | `tmp21 = tl_math.sin(tmp20)` | sin  → y step 21/32 |
| 23 | - | `tmp22 = libdevice.tanh(tmp21)` | tanh  → y step 22/32 |
| 24 | - | `tmp23 = tl_math.sin(tmp22)` | sin  → y step 23/32 |
| 25 | - | `tmp24 = libdevice.tanh(tmp23)` | tanh  → y step 24/32 |
| 26 | - | `tmp25 = tl_math.sin(tmp24)` | sin  → y step 25/32 |
| 27 | - | `tmp26 = libdevice.tanh(tmp25)` | tanh  → y step 26/32 |
| 28 | - | `tmp27 = tl_math.sin(tmp26)` | sin  → y step 27/32 |
| 29 | - | `tmp28 = libdevice.tanh(tmp27)` | tanh  → y step 28/32 |
| 30 | - | `tmp29 = tl_math.sin(tmp28)` | sin  → y step 29/32 |
| 31 | - | `tmp30 = libdevice.tanh(tmp29)` | tanh  → y step 30/32 |
| 32 | - | `tmp31 = tl_math.sin(tmp30)` | sin  → y step 31/32 |
| 33 | - | `tmp32 = libdevice.tanh(tmp31)` | tanh  → y step 32/32  (= y) |
| 34 | - | `tl.store(out_ptr0 + (x0), tmp32, None)` | STORE y step 32/32  (= y) → memory [y] |

**Launch 2: `triton_red_fused_mul_sum_0`** (looped row reduction; group 2: inputs y → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 2 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 2 over the row starts (loop over row chunks) |
| 7 | 2 | `tmp4 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 8 | 2 | `tmp5 = tmp4 * tmp2` | multiply y × row sum Σy → out |
| 9 | 2 | `tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp5, r0_mask)` | STORE out = y × Σy → memory [out] |

### Separate (3 groups) — 3 kernel launch(es), 32 sin/tanh per element

| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |
|---|---|---|---|---|---|---|---|---|
| 1 | group 1 | `triton_poi_fused_sin_tanh_0` | pointwise (1 pass over elements) | 1 | x | y | 32 | 2801.1 |
| 2 | group 2 | `triton_red_fused_sum_0` | looped row reduction | 1 | y | s | 0 | 451.4 |
| 3 | group 3 | `triton_poi_fused_mul_0` | pointwise (1 pass over elements) | 1 | s, y | out | 0 | 934.1 |

**Launch 1: `triton_poi_fused_sin_tanh_0`** (pointwise (1 pass over elements); group 1: inputs x → output y)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x0), None)` | LOAD x ← memory [x] |
| 2 | - | `tmp1 = tl_math.sin(tmp0)` | sin  → y step 1/32 |
| 3 | - | `tmp2 = libdevice.tanh(tmp1)` | tanh  → y step 2/32 |
| 4 | - | `tmp3 = tl_math.sin(tmp2)` | sin  → y step 3/32 |
| 5 | - | `tmp4 = libdevice.tanh(tmp3)` | tanh  → y step 4/32 |
| 6 | - | `tmp5 = tl_math.sin(tmp4)` | sin  → y step 5/32 |
| 7 | - | `tmp6 = libdevice.tanh(tmp5)` | tanh  → y step 6/32 |
| 8 | - | `tmp7 = tl_math.sin(tmp6)` | sin  → y step 7/32 |
| 9 | - | `tmp8 = libdevice.tanh(tmp7)` | tanh  → y step 8/32 |
| 10 | - | `tmp9 = tl_math.sin(tmp8)` | sin  → y step 9/32 |
| 11 | - | `tmp10 = libdevice.tanh(tmp9)` | tanh  → y step 10/32 |
| 12 | - | `tmp11 = tl_math.sin(tmp10)` | sin  → y step 11/32 |
| 13 | - | `tmp12 = libdevice.tanh(tmp11)` | tanh  → y step 12/32 |
| 14 | - | `tmp13 = tl_math.sin(tmp12)` | sin  → y step 13/32 |
| 15 | - | `tmp14 = libdevice.tanh(tmp13)` | tanh  → y step 14/32 |
| 16 | - | `tmp15 = tl_math.sin(tmp14)` | sin  → y step 15/32 |
| 17 | - | `tmp16 = libdevice.tanh(tmp15)` | tanh  → y step 16/32 |
| 18 | - | `tmp17 = tl_math.sin(tmp16)` | sin  → y step 17/32 |
| 19 | - | `tmp18 = libdevice.tanh(tmp17)` | tanh  → y step 18/32 |
| 20 | - | `tmp19 = tl_math.sin(tmp18)` | sin  → y step 19/32 |
| 21 | - | `tmp20 = libdevice.tanh(tmp19)` | tanh  → y step 20/32 |
| 22 | - | `tmp21 = tl_math.sin(tmp20)` | sin  → y step 21/32 |
| 23 | - | `tmp22 = libdevice.tanh(tmp21)` | tanh  → y step 22/32 |
| 24 | - | `tmp23 = tl_math.sin(tmp22)` | sin  → y step 23/32 |
| 25 | - | `tmp24 = libdevice.tanh(tmp23)` | tanh  → y step 24/32 |
| 26 | - | `tmp25 = tl_math.sin(tmp24)` | sin  → y step 25/32 |
| 27 | - | `tmp26 = libdevice.tanh(tmp25)` | tanh  → y step 26/32 |
| 28 | - | `tmp27 = tl_math.sin(tmp26)` | sin  → y step 27/32 |
| 29 | - | `tmp28 = libdevice.tanh(tmp27)` | tanh  → y step 28/32 |
| 30 | - | `tmp29 = tl_math.sin(tmp28)` | sin  → y step 29/32 |
| 31 | - | `tmp30 = libdevice.tanh(tmp29)` | tanh  → y step 30/32 |
| 32 | - | `tmp31 = tl_math.sin(tmp30)` | sin  → y step 31/32 |
| 33 | - | `tmp32 = libdevice.tanh(tmp31)` | tanh  → y step 32/32  (= y) |
| 34 | - | `tl.store(out_ptr0 + (x0), tmp32, None)` | STORE y step 32/32  (= y) → memory [y] |

**Launch 2: `triton_red_fused_sum_0`** (looped row reduction; group 2: inputs y → output s)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `_tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)` | initialise running row sum = 0 |
| 2 | 1 | `for r0_offset in tl.range(0, r0_numel, R0_BLOCK):` | ── pass 1 over the row starts (loop over row chunks) |
| 3 | 1 | `tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)` | LOAD y ← memory [y] |
| 4 | 1 | `tmp3 = _tmp2 + tmp1` | add y to running row sum |
| 5 | end of 1 | `tmp2 = tl.sum(_tmp2, 1)[:, None]` | finish row sum → Σy (one number per row) |
| 6 | 1 | `tl.store(out_ptr0 + (x0), tmp2, None)` | STORE row sum Σy → memory [s] |

**Launch 3: `triton_poi_fused_mul_0`** (pointwise (1 pass over elements); group 3: inputs y, s → output out)

| # | Pass | Generated code (exact) | What it computes |
|---|---|---|---|
| 1 | - | `tmp0 = tl.load(in_ptr0 + (x2), None)` | LOAD y ← memory [y] |
| 2 | - | `tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')` | LOAD s ← memory [s] |
| 3 | - | `tmp2 = tmp0 * tmp1` | multiply y × s → out |
| 4 | - | `tl.store(out_ptr0 + (x2), tmp2, None)` | STORE out = y × Σy → memory [out] |
