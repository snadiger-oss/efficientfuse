"""Read-only software/device provenance, with explicit missing capabilities."""

from importlib import metadata
import os
import platform
import sys

from .energy import NvmlEnergyMeter


def environment_report(device_name="cuda:0", *, exclusive_gpu=False):
    report = {"python": sys.version, "platform": platform.platform(), "device_requested": device_name,
              "versions": {}, "status": "unavailable", "reason": None,
              "environment": {key: os.environ.get(key) for key in (
                  "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING", "OMP_NUM_THREADS",
                  "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "SLURM_JOB_ID")}}
    for package in ("torch", "triton", "nvidia-ml-py"):
        try:
            report["versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["versions"][package] = None
    meter = None
    try:
        import torch
        device = torch.device(device_name)
        report.update(cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                      torch_threads=torch.get_num_threads(), float32_matmul_precision=torch.get_float32_matmul_precision(),
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      cudnn_benchmark=torch.backends.cudnn.benchmark,
                      cudnn_deterministic=torch.backends.cudnn.deterministic)
        if device.type == "cpu":
            report.update(status="cpu_development", reason="CPU timing only; no GPU energy or GPU performance claims")
            return report
        if device.type != "cuda" or not torch.cuda.is_available() or torch.version.cuda is None:
            raise RuntimeError("NVIDIA CUDA is unavailable; no CPU fallback")
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        report["device"] = {"logical_index": index, "name": props.name, "total_memory_bytes": props.total_memory,
                            "compute_capability": [props.major, props.minor], "uuid": str(getattr(props, "uuid", None))}
        meter = NvmlEnergyMeter.open(torch.device(f"cuda:{index}"), exclusive_gpu=exclusive_gpu)
        state = meter.snapshot()
        report["energy"] = {"counter_available": meter.available, "reason": meter.qualification_reason(state),
                            "identity": meter.identity, "state": state,
                            "scope": "Whole GPU-device energy; includes idle/background energy within counter boundaries"}
        report.update(status="cuda_available", reason=None)
    except Exception as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    finally:
        if meter is not None:
            meter.close()
    return report
