# AOT ID: ['12_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_havarur/tu/ctuglk7mtktzb3xyu63fz3gqsrqbojbv2ggfbohrshjkxuyd2hwe.py
# Topologically Sorted Source Nodes: [y, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11, sum_1, mul], Original ATen: [aten.sin, aten.tanh, aten.sum, aten.mul]
# Source node to ATen node mapping:
#   mul => mul
#   sum_1 => sum_1
#   y => sin
#   y_1 => tanh
#   y_10 => sin_5
#   y_11 => tanh_5
#   y_2 => sin_1
#   y_3 => tanh_1
#   y_4 => sin_2
#   y_5 => tanh_2
#   y_6 => sin_3
#   y_7 => tanh_3
#   y_8 => sin_4
#   y_9 => tanh_4
# Graph fragment:
#   %arg0_1 : Tensor "f32[8192, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sum_1 : Tensor "f32[8192, 1][1, 8192]cuda:0" = PlaceHolder[target=sum_1]
#   %sin : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%arg0_1,), kwargs = {})
#   %tanh : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin,), kwargs = {})
#   %sin_1 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh,), kwargs = {})
#   %tanh_1 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_1,), kwargs = {})
#   %sin_2 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_1,), kwargs = {})
#   %tanh_2 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_2,), kwargs = {})
#   %sin_3 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_2,), kwargs = {})
#   %tanh_3 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_3,), kwargs = {})
#   %sin_4 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_3,), kwargs = {})
#   %tanh_4 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_4,), kwargs = {})
#   %sin_5 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_4,), kwargs = {})
#   %tanh_5 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_5,), kwargs = {})
#   %sum_1 : Tensor "f32[8192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tanh_5, [-1], True), kwargs = {})
#   %mul : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_5, %sum_1), kwargs = {})
#   return %sum_1,%mul
triton_red_fused_mul_sin_sum_tanh_0 = async_compile.triton('triton_red_fused_mul_sin_sum_tanh_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_mul_sin_sum_tanh_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'x': 0, 'r0_': 805306368}, 'add_persistent_rblock': True, 'backend_hash': '3BF43BD63B64A2C856259672BEB723B124DEA5263CBA4F3EDDA0CEDF631CFDEA', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_mul_sin_sum_tanh_0(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl_math.sin(tmp0)
        tmp2 = libdevice.tanh(tmp1)
        tmp3 = tl_math.sin(tmp2)
        tmp4 = libdevice.tanh(tmp3)
        tmp5 = tl_math.sin(tmp4)
        tmp6 = libdevice.tanh(tmp5)
        tmp7 = tl_math.sin(tmp6)
        tmp8 = libdevice.tanh(tmp7)
        tmp9 = tl_math.sin(tmp8)
        tmp10 = libdevice.tanh(tmp9)
        tmp11 = tl_math.sin(tmp10)
        tmp12 = libdevice.tanh(tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp16 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl_math.sin(tmp16)
        tmp18 = libdevice.tanh(tmp17)
        tmp19 = tl_math.sin(tmp18)
        tmp20 = libdevice.tanh(tmp19)
        tmp21 = tl_math.sin(tmp20)
        tmp22 = libdevice.tanh(tmp21)
        tmp23 = tl_math.sin(tmp22)
        tmp24 = libdevice.tanh(tmp23)
        tmp25 = tl_math.sin(tmp24)
        tmp26 = libdevice.tanh(tmp25)
        tmp27 = tl_math.sin(tmp26)
        tmp28 = libdevice.tanh(tmp27)
        tmp29 = tmp28 * tmp14
        tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp29, r0_mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (8192, 8192), (8192, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf1 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
            # Topologically Sorted Source Nodes: [y, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11, sum_1, mul], Original ATen: [aten.sin, aten.tanh, aten.sum, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_mul_sin_sum_tanh_0.run(arg0_1, buf1, 8192, 8192, stream=raw_stream0)
            del arg0_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
