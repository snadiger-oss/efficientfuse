"""Reference checks performed during setup, outside measurement windows."""

import math

from greenfuse.execution.buffers import synchronize


def check_correctness(executor, reference, inputs, *, rtol=1e-5, atol=1e-6):
    """Check original and changed values, output metadata, and input mutation."""
    import torch

    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (rtol, atol)):
        raise ValueError("Correctness tolerances must be finite and nonnegative")
    with torch.inference_mode(False), torch.no_grad():
        changed = {name: value.neg().add(0.25) for name, value in inputs.items()}
    observations = []
    with torch.inference_mode():
        for label, batch in (("original", inputs), ("changed", changed)):
            before = {name: value.clone() for name, value in batch.items()}
            # Independent copies prevent a mutating reference from changing the
            # candidate's inputs or masking an incorrect implementation.
            expected = reference(**{name: value.clone() for name, value in batch.items()})
            actual = executor.run(batch)
            synchronize(executor.device)
            if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
                raise AssertionError("This measurement contract requires one tensor output")
            if not torch.isfinite(expected).all().item() or not torch.isfinite(actual).all().item():
                raise AssertionError("Nonfinite correctness output")
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            for name, value in batch.items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            observations.append({"inputs": label, "max_absolute_error": (actual - expected).abs().max().item()})
    return {"status": "passed", "rtol": rtol, "atol": atol, "checks": observations}
