"""Device management utilities"""

import torch

def get_device(device_str: str = None) -> torch.device:
    """Get torch device"""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
