"""Strongly Entangling Layers"""

import torch
import torch.nn as nn
from typing import Optional

from ..core.module import QuantumModule
from ..core.parameter import Parameter
from ..core.qvector import qvector
from ..gates.parametric import u3
from ..gates.two_qubit import cx

class StronglyEntangling(QuantumModule):
    """
    Strongly Entangling Layers
    
    A multi-layer variational ansatz consisting of:
    1. Single qubit rotations (U3) on all qubits
    2. Entanglers (CNOTs) arranged in a specific pattern (e.g. circular)
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        ranges: List of ranges for CNOTs per layer (optional)
        imprimitive: Two-qubit gate to use (default: CX) - currently only CX supported effectively
    """
    def __init__(
        self, 
        n_qubits: int, 
        n_layers: int = 1,
        ranges: Optional[list] = None
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        if ranges is None:
            # Default ranges: [1, 2, 4, ..., 2^(n_layers-1) mod n_qubits]
            # Standard "Strongly Entangling" usually increases range
            self.ranges = [r % n_qubits for r in range(1, n_layers + 1)] # Simple increasing range
            # Or simplified: [1] * n_layers for nearest neighbor
        else:
            self.ranges = ranges
            
        # Parameters: [n_layers, n_qubits, 3] for U3(theta, phi, lam)
        self.weights = Parameter(
            shape=(n_layers, n_qubits, 3),
            init_method="uniform",
            init_range=(-3.14, 3.14)
        )
        
    def forward(self, q: qvector):
        """
        Apply layers to the qvector
        """
        if q.n_qubits != self.n_qubits:
            raise ValueError(f"QVector has {q.n_qubits} qubits, but layer expects {self.n_qubits}")
            
        for l in range(self.n_layers):
            # 1. Rotations (U3)
            # weights[l, i, :] -> [3]
            # But specific gate calls need scalar or batch of scalars.
            # U3 implementation in gates/parametric.py assumes (theta, phi, lam, target)
            # We need to broadcast properly if qvector has batch.
            
            # Since qvector is generic, we just record operations.
            # But the 'weights' is a logical parameter tensor.
            # We iterate over qubits.
            
            w_layer = self.weights[l] # [n_qubits, 3]
            
            for i in range(self.n_qubits):
                # Apply U3
                # u3(theta, phi, lam, target)
                # w_layer[i, 0], w_layer[i, 1], w_layer[i, 2]
                u3(w_layer[i, 0], w_layer[i, 1], w_layer[i, 2], q[i])
                
            # 2. Entanglers
            r = self.ranges[l % len(self.ranges)]
            if r == 0: r = 1 # Avoid self-loop if logic flaked
            
            for i in range(self.n_qubits):
                control = i
                target = (i + r) % self.n_qubits
                cx(q[control], q[target])
                
        return q
