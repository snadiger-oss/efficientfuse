"""Create and validate inputs outside execution/measurement loops."""

from .specs import WorkloadSpec


def make_inputs(spec: WorkloadSpec, *, device: str = "cpu") -> dict:
    import torch

    # A private CPU generator leaves the caller's global RNG state unchanged.
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    inputs = {
        item.name: torch.randn(item.shape, generator=generator, dtype=torch.float32).to(device)
        for item in spec.inputs
    }
    validate_inputs(spec, inputs)
    return inputs


def validate_inputs(spec: WorkloadSpec, inputs: dict) -> None:
    import torch

    expected = {item.name for item in spec.inputs}
    if set(inputs) != expected:
        raise ValueError(f"Expected input names {sorted(expected)}, got {sorted(inputs)}")
    devices = set()
    for item in spec.inputs:
        tensor = inputs[item.name]
        if type(tensor) is not torch.Tensor:
            raise ValueError(f"{item.name}: expected a plain torch.Tensor")
        if tuple(tensor.shape) != item.shape:
            raise ValueError(f"{item.name}: expected shape {item.shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError(f"{item.name}: expected contiguous float32")
        if tensor.requires_grad:
            raise ValueError(f"{item.name}: the initial contract is forward-only")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{item.name}: expected finite input values")
        devices.add(tensor.device)
    if len(devices) != 1:
        raise ValueError("All inputs must be on the same device")
