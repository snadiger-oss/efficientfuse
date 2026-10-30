"""Fixed-input contract and stable buffers. PyTorch is imported only on use."""

from dataclasses import asdict, dataclass


class UnsupportedExecutionError(RuntimeError):
    """The requested device or execution mode is unavailable."""


@dataclass(frozen=True)
class TensorSignature:
    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class InputContract:
    tensors: tuple[TensorSignature, ...]

    @classmethod
    def from_inputs(cls, inputs: dict, names: tuple[str, ...], *, allow_cpu=False):
        import torch

        if set(inputs) != set(names) or not names:
            raise ValueError(f"Expected exactly these input names: {names}")
        signatures = []
        for name in names:
            value = inputs[name]
            if type(value) is not torch.Tensor:
                raise ValueError(f"{name}: expected a plain torch.Tensor")
            if value.device.type not in (("cpu", "cuda") if allow_cpu else ("cuda",)):
                raise UnsupportedExecutionError("CUDA tensors are required; CPU development must be explicitly enabled")
            if value.device.type == "cuda" and torch.version.cuda is None:
                raise UnsupportedExecutionError("This experiment requires NVIDIA CUDA, not another accelerator backend")
            if value.dtype != torch.float32 or not value.is_contiguous() or value.numel() == 0:
                raise ValueError(f"{name}: expected nonempty contiguous float32")
            if value.requires_grad or value.is_inference() or value.storage_offset() != 0:
                raise ValueError(f"{name}: require ordinary, zero-offset tensors without gradients")
            signatures.append(TensorSignature(name, tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device)))
        if len({item.device for item in signatures}) != 1:
            raise ValueError("All inputs must be on the same device")
        contract = cls(tuple(signatures))
        contract.validate(inputs)
        return contract

    @property
    def device(self):
        import torch

        return torch.device(self.tensors[0].device)

    def validate(self, inputs: dict) -> tuple:
        import torch

        names = tuple(item.name for item in self.tensors)
        if set(inputs) != set(names):
            raise ValueError(f"Expected exactly these input names: {names}")
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError("Autocast must be disabled for the fixed FP32 execution contract")
        ordered = []
        for spec in self.tensors:
            value = inputs[spec.name]
            if type(value) is not torch.Tensor:
                raise ValueError(f"{spec.name}: expected a plain torch.Tensor")
            actual = (tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
            expected = (spec.shape, spec.stride, spec.dtype, spec.device)
            if actual != expected or value.requires_grad or value.is_inference() or value.storage_offset() != 0:
                raise ValueError(f"{spec.name}: shape/stride/dtype/device or tensor policy changed; prepare again")
            ordered.append(value)
        return tuple(ordered)

    def to_dict(self):
        return {
            "tensors": [asdict(item) for item in self.tensors],
            "forward_only": True,
            "autocast": False,
            "output_lifetime": "borrowed until the next run or close; clone to retain",
            "concurrency": "one caller, on the CUDA stream used during preparation",
        }


def synchronize(device):
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def current_stream_id(device):
    if device.type == "cuda":
        import torch

        return torch.cuda.current_stream(device).cuda_stream
    return None


def check_stream(device, stream_id):
    if current_stream_id(device) != stream_id:
        raise ValueError("Run on the CUDA stream used during preparation; concurrent/cross-stream use is unsupported")


class StaticBuffers:
    """Own input memory at fixed addresses. Every graph run copies new inputs."""

    def __init__(self, contract: InputContract, inputs: dict):
        import torch

        self.contract = contract
        values = contract.validate(inputs)
        # Preserve ordinary input tensor semantics to avoid new compile guards
        # when capture follows compilation with ordinary example inputs.
        with torch.inference_mode(False), torch.no_grad():
            self.values = tuple(value.detach().clone() for value in values)
        self.addresses = tuple(value.data_ptr() for value in self.values)

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in self.values)

    def update(self, inputs: dict):
        import torch

        incoming = self.contract.validate(inputs)
        with torch.no_grad():
            for target, source in zip(self.values, incoming):
                target.copy_(source, non_blocking=True)

    def close(self):
        self.values = ()
