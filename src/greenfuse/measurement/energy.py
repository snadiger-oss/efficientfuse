"""NVML cumulative GPU energy; no instantaneous-power/TDP fallback.

Counter units and device identity follow NVIDIA's NVML Device Queries reference:
https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html
"""

import math
import os
from time import perf_counter
import uuid

from greenfuse.execution.buffers import synchronize
from greenfuse.results import Trial


def physical_uuid(value):
    """Normalize a full CUDA/NVML UUID, rejecting MIG and ambiguous identities."""
    if isinstance(value, bytes):
        value = str(uuid.UUID(bytes=value)) if len(value) == 16 else value.decode("ascii")
    text = str(value)
    if text.upper().startswith("MIG-"):
        raise ValueError("MIG slices cannot be attributed whole-device energy")
    if text.upper().startswith("GPU-"):
        text = text[4:]
    return "GPU-" + str(uuid.UUID(text))


def energy_per_execution(start_mj, end_mj, completed):
    if any(type(v) is not int or v < 0 for v in (start_mj, end_mj)):
        raise ValueError("Energy counters must be nonnegative integers in millijoules")
    if type(completed) is not int or completed <= 0:
        raise ValueError("completed must be positive")
    if end_mj <= start_mj:
        raise ValueError("Energy counter did not increase (reset, unsupported resolution, or too short a window)")
    return (end_mj - start_mj) / 1000 / completed


class NvmlEnergyMeter:
    """One read-only NVML session, tied to the UUID reported by PyTorch.

exclusive_gpu is the operator's assertion of a whole physical GPU reservation.
Boundary process checks supplement that assertion; they cannot detect every
short-lived concurrent process. No clocks, power limits or processes are changed.
"""

    def __init__(self, *, exclusive_gpu=False):
        self.exclusive_gpu = exclusive_gpu
        self.available = False
        self.reason = "Not initialized"
        self.identity = {}
        self.nvml = None
        self.handle = None
        self.initialized = False

    @classmethod
    def open(cls, device, *, exclusive_gpu=False):
        meter = cls(exclusive_gpu=exclusive_gpu)
        try:
            import torch
            if device.type != "cuda" or not torch.cuda.is_available() or torch.version.cuda is None:
                raise RuntimeError("An NVIDIA CUDA GPU is required for GPU energy")
            properties = torch.cuda.get_device_properties(device)
            gpu_uuid = physical_uuid(getattr(properties, "uuid", None))
            import pynvml

            meter.connect(pynvml, gpu_uuid)
        except Exception as error:
            meter.available = False
            meter.reason = f"{type(error).__name__}: {error}"
        return meter

    def connect(self, nvml, cuda_uuid):
        """Separate connection path permits labeled fake-NVML software tests."""
        self.nvml = nvml
        expected = physical_uuid(cuda_uuid)
        nvml.nvmlInit()
        self.initialized = True
        self.handle = nvml.nvmlDeviceGetHandleByUUID(expected)
        observed = physical_uuid(nvml.nvmlDeviceGetUUID(self.handle))
        if observed != expected:
            raise ValueError("PyTorch UUID and NVML UUID differ; refusing device-index guessing")
        self.identity = {"cuda_uuid": expected, "nvml_uuid": observed, "mapping": "exact UUID match",
                         "gpu_name": self._text(nvml.nvmlDeviceGetName(self.handle)),
                         "driver_version": self._text(nvml.nvmlSystemGetDriverVersion()),
                         "pci_bus_id": self._text(nvml.nvmlDeviceGetPciInfo(self.handle).busId),
                         "exclusive_physical_gpu_asserted": self.exclusive_gpu}
        # NotSupported on older non-MIG devices is acceptable. Permission and
        # other failures do not establish that the GPU is unsliced.
        try:
            mode, _ = nvml.nvmlDeviceGetMigMode(self.handle)
            if mode != 0:
                raise ValueError("MIG is enabled; whole-device energy attribution is unsupported")
        except nvml.NVMLError_NotSupported:
            pass
        try:
            if nvml.nvmlDeviceGetVirtualizationMode(self.handle) != 0:
                raise ValueError("Virtualized GPU energy attribution is not supported in this experiment")
        except nvml.NVMLError_NotSupported:
            pass
        self.read_mj()
        self.available, self.reason = True, None

    @staticmethod
    def _text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    def read_mj(self):
        value = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        if type(value) is not int or value < 0:
            raise ValueError("NVML did not return a nonnegative integer counter")
        return value

    def snapshot(self):
        if self.handle is None:
            return {"status": "unavailable", "reason": self.reason}
        nvml, handle = self.nvml, self.handle
        result = {"errors": {}}
        readers = {
            "temperature_c": lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
            "power_limit_mw": lambda: nvml.nvmlDeviceGetPowerManagementLimit(handle),
            "graphics_clock_mhz": lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS),
            "memory_clock_mhz": lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM),
            "memory_used_bytes": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).used,
            "gpu_utilization_percent": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "performance_state": lambda: nvml.nvmlDeviceGetPerformanceState(handle),
            "clock_throttle_reasons": lambda: nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
            "application_graphics_clock_mhz": lambda: nvml.nvmlDeviceGetApplicationsClock(handle, nvml.NVML_CLOCK_GRAPHICS),
            "application_memory_clock_mhz": lambda: nvml.nvmlDeviceGetApplicationsClock(handle, nvml.NVML_CLOCK_MEM),
            "compute_pids": lambda: sorted({p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)}),
            "graphics_pids": lambda: sorted({p.pid for p in nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)}),
        }
        for name, reader in readers.items():
            try:
                result[name] = reader()
            except Exception as error:
                result[name] = None
                result["errors"][name] = f"{type(error).__name__}: {error}"
        return result

    def qualification_reason(self, snapshot):
        if not self.available:
            return self.reason
        if not self.exclusive_gpu:
            return "Whole physical GPU reservation was not asserted; use --exclusive-gpu only during an exclusive allocation"
        if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") or os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
            return "CUDA MPS environment detected; shared-device energy is unsupported"
        for name in ("compute_pids", "graphics_pids"):
            if snapshot.get(name) is None:
                return f"Cannot check concurrent GPU processes: {name} unavailable"
            others = set(snapshot[name]) - {os.getpid()}
            if others:
                return f"Concurrent GPU processes detected: {sorted(others)}"
        return None

    def close(self):
        if self.initialized:
            self.nvml.nvmlShutdown()
            self.initialized = False


def measure_energy(executor, inputs, iterations, meter, *, sync=None, clock=perf_counter):
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    trial = Trial("energy", iterations)
    sync = sync or (lambda: synchronize(executor.device))
    try:
        trial.environment_before = meter.snapshot()
        reason = meter.qualification_reason(trial.environment_before)
        if reason:
            trial.status, trial.reason = "unavailable", reason
            return trial
        sync()
        trial.start_monotonic_seconds = clock()
        trial.energy_start_mj = meter.read_mj()
        trial.counter_read_intervals.append([trial.start_monotonic_seconds, clock()])
        for _ in range(iterations):
            executor.run(inputs)
            trial.submitted_iterations += 1
        sync()
        trial.completed_iterations = iterations
        end_read_start = clock()
        trial.energy_end_mj = meter.read_mj()
        trial.end_monotonic_seconds = clock()
        trial.counter_read_intervals.append([end_read_start, trial.end_monotonic_seconds])
        trial.elapsed_seconds = trial.end_monotonic_seconds - trial.start_monotonic_seconds
        if not math.isfinite(trial.elapsed_seconds) or trial.elapsed_seconds <= 0:
            raise ValueError("Invalid energy window duration")
        trial.environment_after = meter.snapshot()
        reason = meter.qualification_reason(trial.environment_after)
        if reason:
            trial.status, trial.reason = "invalid", reason
            return trial
        try:
            trial.energy_joules_per_execution = energy_per_execution(trial.energy_start_mj, trial.energy_end_mj, iterations)
            trial.status = "ok"
        except ValueError as error:
            trial.status, trial.reason = "invalid", str(error)
    except Exception as error:
        trial.status, trial.reason = "failed", f"{type(error).__name__}: {error}"
        trial.energy_joules_per_execution = None
    return trial
