"""Write each original workload once; partitions are generated elsewhere."""


def workload_w1(x, bias, residual):
    """Bias addition, ReLU, and residual addition; also the eager reference."""
    import torch

    return torch.relu(x + bias) + residual


def workload_w2(x, weight, bias):
    """Linear + GELU as three operations: matmul, bias add, GELU (exact erf form).

    Fusion here can change the GEMM implementation itself (cuBLAS vs a Triton
    template with a fused epilogue), not just memory traffic.
    """
    import torch.nn.functional as functional

    return functional.gelu(x @ weight + bias)


def make_workload_w3(n: int):
    """Residual add, LayerNorm over the last dimension (no affine), GELU.

    LayerNorm needs a static normalized_shape, so the function is built per row width N.
    The reduction is where fusing neighbors changes how work is split across the GPU.
    """

    def workload_w3(x, residual):
        import torch.nn.functional as functional

        return functional.gelu(functional.layer_norm(x + residual, (n,)))

    return workload_w3


def workload_w4(x, bias):
    """Fan-out: h = x + bias feeds both a row mean and the final subtraction (mean-centering).

    Fusing everything can make the compiler reload or recompute h for the second use;
    splitting after h stores it once and reads it twice.
    """
    h = x + bias
    return h - h.mean(dim=-1, keepdim=True)
