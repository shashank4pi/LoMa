"""Device/autocast-dtype defaults for LoMa.

Callers that don't pass an explicit device to a LoMa constructor can use
`default_device_and_dtype()` to get the same hardware-auto-detect behaviour
the library used to have, but without any module-level side effects.

If the caller pins the device but not the dtype (or vice-versa), use
`default_amp_dtype_for(device)` to pick the dtype that matches that device.
"""
import torch


def default_amp_dtype_for(device: torch.device) -> torch.dtype:
    """Autocast dtype that matches ``device``."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return (
            torch.bfloat16
            if torch.cuda.get_device_capability(idx)[0] >= 8
            else torch.float16
        )
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def default_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        return dev, default_amp_dtype_for(dev)
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        return dev, default_amp_dtype_for(dev)
    dev = torch.device("cpu")
    return dev, default_amp_dtype_for(dev)
