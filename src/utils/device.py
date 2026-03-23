import torch


def get_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device() -> torch.device:
    return torch.device(get_device_type())
