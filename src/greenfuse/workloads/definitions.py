"""Write each original workload once; partitions are generated elsewhere."""


def workload_w1(x, bias, residual):
    """Bias addition, ReLU, and residual addition; also the eager reference."""
    import torch

    return torch.relu(x + bias) + residual
