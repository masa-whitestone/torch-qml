"""Quantum module base class"""

import torch
import torch.nn as nn
from typing import Optional

from ..backends.config import get_backend


class QuantumModule(nn.Module):
    """
    Base class for quantum circuits
    
    Inherits from nn.Module and adds functionality specific to quantum circuits
    """
    
    def __init__(self, backend: Optional[str] = None):
        super().__init__()
        self._backend = backend or get_backend()
        self._device = None
    
    @property
    def device(self) -> torch.device:
        """Return current device"""
        if self._device is not None:
            return self._device
        
        # Estimate device from parameters
        for param in self.parameters():
            return param.device
        
        # Default
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def to(self, device):
        """Set device"""
        self._device = torch.device(device)
        return super().to(device)
