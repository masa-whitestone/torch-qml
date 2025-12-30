"""
torchqml - PyTorch Quantum Machine Learning with cuQuantum

Fast Adjoint Differentiation implementation with cuQuantum backend.
"""

__version__ = "0.1.0"

# Circuit
from .circuit import QuantumCircuit

# Operators
from .operators import X, Y, Z, I, PauliOperator

# Gates (for direct import)
from .gates import (
    H, X as XGate, Y as YGate, Z as ZGate, S, T,
    RX, RY, RZ,
    CNOT, CX, CZ, SWAP
)

# nn.Module layers
from .nn import (
    QuantumLayer,
    HybridQuantumLayer,
    StronglyEntanglingLayer
)

# Autograd
from .autograd import QuantumExpectation, quantum_expectation

# Functional API
from . import functional

__all__ = [
    # Version
    "__version__",
    
    # Circuit
    "QuantumCircuit",
    
    # Operators
    "X", "Y", "Z", "I", "PauliOperator",
    
    # Gates
    "H", "XGate", "YGate", "ZGate", "S", "T",
    "RX", "RY", "RZ",
    "CNOT", "CX", "CZ", "SWAP",
    
    # Layers
    "QuantumLayer",
    "HybridQuantumLayer", 
    "StronglyEntanglingLayer",
    
    # Autograd
    "QuantumExpectation",
    "quantum_expectation",
    
    # Functional
    "functional",
]
