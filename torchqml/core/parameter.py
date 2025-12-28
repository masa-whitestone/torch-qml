"""Quantum circuit parameter"""

import torch
import torch.nn as nn
from typing import Tuple, Union, Optional
import math


class Parameter(nn.Parameter):
    """
    Quantum circuit parameter
    
    Wrapper for nn.Parameter with additional initialization options
    """
    
    def __new__(
        cls,
        shape: Union[Tuple[int, ...], int],
        init: str = "uniform",
        requires_grad: bool = True,
        **kwargs
    ):
        if isinstance(shape, int):
            shape = (shape,)
        
        # Initialization
        if init == "zeros":
            data = torch.zeros(shape)
        elif init == "ones":
            data = torch.ones(shape)
        elif init == "uniform":
            # [-pi, pi] uniform distribution
            data = torch.empty(shape).uniform_(-math.pi, math.pi)
        elif init == "normal":
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 0.1)
            data = torch.empty(shape).normal_(mean, std)
        elif init == "xavier":
            data = torch.empty(shape)
            nn.init.xavier_uniform_(data.view(-1, 1) if data.dim() == 1 else data)
        else:
            raise ValueError(f"Unknown initialization: {init}")
        
        instance = super().__new__(cls, data, requires_grad=requires_grad)
        return instance
