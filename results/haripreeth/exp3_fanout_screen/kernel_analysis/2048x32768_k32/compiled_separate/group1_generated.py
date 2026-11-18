# AOT ID: ['67_inference']
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


# kernel path: /tmp/torchinductor_havarur/6p/c6piuroecgc5hiwfjufnlxrdgkv23pgop4ppkauba2m3nvwjohgj.py
# Topologically Sorted Source Nodes: [y, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18, y_19, y_20, y_21, y_22, y_23, y_24, y_25, y_26, y_27, y_28, y_29, y_30, y_31], Original ATen: [aten.sin, aten.tanh]
# Source node to ATen node mapping:
#   y => sin
#   y_1 => tanh
#   y_10 => sin_5
#   y_11 => tanh_5
#   y_12 => sin_6
#   y_13 => tanh_6
#   y_14 => sin_7
#   y_15 => tanh_7
#   y_16 => sin_8
#   y_17 => tanh_8
#   y_18 => sin_9
#   y_19 => tanh_9
#   y_2 => sin_1
#   y_20 => sin_10
#   y_21 => tanh_10
#   y_22 => sin_11
#   y_23 => tanh_11
#   y_24 => sin_12
#   y_25 => tanh_12
#   y_26 => sin_13
#   y_27 => tanh_13
#   y_28 => sin_14
#   y_29 => tanh_14
#   y_3 => tanh_1
#   y_30 => sin_15
#   y_31 => tanh_15
#   y_4 => sin_2
#   y_5 => tanh_2
#   y_6 => sin_3
#   y_7 => tanh_3
#   y_8 => sin_4
#   y_9 => tanh_4
# Graph fragment:
#   %arg0_1 : Tensor "f32[2048, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sin : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%arg0_1,), kwargs = {})
#   %tanh : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin,), kwargs = {})
#   %sin_1 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh,), kwargs = {})
#   %tanh_1 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_1,), kwargs = {})
#   %sin_2 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_1,), kwargs = {})
#   %tanh_2 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_2,), kwargs = {})
#   %sin_3 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_2,), kwargs = {})
#   %tanh_3 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_3,), kwargs = {})
#   %sin_4 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_3,), kwargs = {})
#   %tanh_4 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_4,), kwargs = {})
#   %sin_5 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_4,), kwargs = {})
#   %tanh_5 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_5,), kwargs = {})
#   %sin_6 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_5,), kwargs = {})
#   %tanh_6 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_6,), kwargs = {})
#   %sin_7 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_6,), kwargs = {})
#   %tanh_7 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_7,), kwargs = {})
#   %sin_8 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_7,), kwargs = {})
#   %tanh_8 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_8,), kwargs = {})
#   %sin_9 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_8,), kwargs = {})
#   %tanh_9 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_9,), kwargs = {})
#   %sin_10 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_9,), kwargs = {})
#   %tanh_10 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_10,), kwargs = {})
#   %sin_11 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_10,), kwargs = {})
#   %tanh_11 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_11,), kwargs = {})
#   %sin_12 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_11,), kwargs = {})
#   %tanh_12 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_12,), kwargs = {})
#   %sin_13 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_12,), kwargs = {})
#   %tanh_13 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_13,), kwargs = {})
#   %sin_14 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_13,), kwargs = {})
#   %tanh_14 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_14,), kwargs = {})
#   %sin_15 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh_14,), kwargs = {})
#   %tanh_15 : Tensor "f32[2048, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sin_15,), kwargs = {})
#   return %tanh_15
triton_poi_fused_sin_tanh_0 = async_compile.triton('triton_poi_fused_sin_tanh_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_sin_tanh_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 805306368}, 'backend_hash': '3BF43BD63B64A2C856259672BEB723B124DEA5263CBA4F3EDDA0CEDF631CFDEA', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sin_tanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
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
    tmp13 = tl_math.sin(tmp12)
    tmp14 = libdevice.tanh(tmp13)
    tmp15 = tl_math.sin(tmp14)
    tmp16 = libdevice.tanh(tmp15)
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
    tmp29 = tl_math.sin(tmp28)
    tmp30 = libdevice.tanh(tmp29)
    tmp31 = tl_math.sin(tmp30)
    tmp32 = libdevice.tanh(tmp31)
    tl.store(out_ptr0 + (x0), tmp32, None)
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
        assert_size_stride(arg0_1, (2048, 32768), (32768, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((2048, 32768), (32768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [y, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18, y_19, y_20, y_21, y_22, y_23, y_24, y_25, y_26, y_27, y_28, y_29, y_30, y_31], Original ATen: [aten.sin, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_sin_tanh_0.run(arg0_1, buf0, 67108864, stream=raw_stream0)
            del arg0_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2048, 32768), (32768, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
