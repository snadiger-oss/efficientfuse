"""Opt-in tests on an actual NVIDIA CUDA GPU."""

import importlib.util
import os


def gpu_tests_enabled():
    if os.environ.get("GREENFUSE_RUN_GPU_TESTS") != "1" or importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


GPU_SKIP_REASON = "Requires NVIDIA CUDA and GREENFUSE_RUN_GPU_TESTS=1; CPU tests are not GPU validation"
