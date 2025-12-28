"""
torchqml - PyTorch-native Quantum Machine Learning with CUDA-Q backend

Usage:
    import torchqml as tq
    
    tq.set_backend("nvidia")
    
    class MyCircuit(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.theta = tq.Parameter((2,))
        
        def forward(self, x):
            q = tq.qvector(2, x.shape[0], device=x.device)
            tq.ry(x[:, 0], q[0])
            tq.rx(self.theta[0], q[0])
            tq.cx(q[0], q[1])
            return tq.expval(q, tq.Z(0))
"""

__version__ = "0.1.0"

# Backend configuration
from .backends.config import (
    set_backend,
    get_backend,
    backend,
    set_random_seed,
)

# Core
from .core.qvector import qvector, QVector
from .core.parameter import Parameter
from .core.module import QuantumModule

# Gates - Single qubit
from .gates.single import h, x, y, z, s, t

# Gates - Parametric
from .gates.parametric import rx, ry, rz, r1, u3

# Gates - Two qubit
from .gates.two_qubit import cx, cy, cz, swap

# Operators
from .operators.pauli import X, Y, Z, I, PauliOperator

# Measurement
from .measure.expval import expval
from .measure.sample import sample
from .measure.state import state

# Layers
from .layers.strongly_entangling import StronglyEntangling

# Utils
from .utils import get_device

__all__ = [
    # Version
    "__version__",
    
    # Backend
    "set_backend",
    "get_backend", 
    "backend",
    "set_random_seed",
    
    # Core
    "qvector",
    "QVector",
    "Parameter",
    "QuantumModule",
    
    # Gates
    "h", "x", "y", "z", "s", "t",
    "rx", "ry", "rz", "r1", "u3",
    "cx", "cy", "cz", "swap",
    
    # Layers
    "StronglyEntangling",
    
    # Operators
    "X", "Y", "Z", "I",
    "PauliOperator",
    
    # Measurement
    "expval",
    "sample",
    "state",
    
    # Utils
    "get_device",
]
