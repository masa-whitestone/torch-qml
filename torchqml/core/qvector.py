"""Quantum state vector"""

import torch
from typing import Union, List, Optional
from ..backends.config import _ensure_initialized


class QVector:
    """
    Class representing a quantum state vector.
    
    Internally, the state is created when the CUDA-Q kernel is executed.
    This class holds metadata for circuit construction.
    """
    
    def __init__(
        self, 
        n_qubits: int, 
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ):
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Record circuit operations (compiled later to CUDA-Q kernel)
        self._operations: List[dict] = []
        
        # State lazy evaluation (calculated at measurement)
        self._state_cache: Optional[torch.Tensor] = None
        self._cache_valid = False
    
    def __getitem__(self, index: Union[int, slice]) -> 'Union[QubitRef, QubitRefSlice]':
        """Return reference to individual qubit(s)"""
        if isinstance(index, int):
            if index < 0:
                index = self.n_qubits + index
            if not 0 <= index < self.n_qubits:
                raise IndexError(f"Qubit index {index} out of range [0, {self.n_qubits})")
            return QubitRef(self, index)
        elif isinstance(index, slice):
            indices = range(*index.indices(self.n_qubits))
            return QubitRefSlice(self, list(indices))
        else:
            raise TypeError(f"Invalid index type: {type(index)}")
    
    def _add_operation(self, op: dict):
        """Add operation and invalidate cache"""
        self._operations.append(op)
        self._cache_valid = False
    
    def _get_operations(self) -> List[dict]:
        """Return list of recorded operations"""
        return self._operations.copy()
    
    def _reset(self):
        """Reset state (start of new forward pass)"""
        self._operations.clear()
        self._cache_valid = False
        self._state_cache = None


class QubitRef:
    """Reference to a single qubit"""
    
    def __init__(self, qvector: QVector, index: int):
        self.qvector = qvector
        self.index = index
    
    @property
    def n_qubits(self) -> int:
        return self.qvector.n_qubits
    
    @property
    def batch_size(self) -> int:
        return self.qvector.batch_size
    
    @property
    def device(self) -> torch.device:
        return self.qvector.device


class QubitRefSlice:
    """Reference to multiple qubits (slice)"""
    
    def __init__(self, qvector: QVector, indices: List[int]):
        self.qvector = qvector
        self.indices = indices
    
    def __iter__(self):
        for i in self.indices:
            yield QubitRef(self.qvector, i)
    
    def __len__(self):
        return len(self.indices)


def qvector(
    n_qubits: int, 
    batch_size: int = 1, 
    device: Optional[torch.device] = None
) -> QVector:
    """
    Create a quantum state vector
    
    Args:
        n_qubits: Number of qubits
        batch_size: Batch size
        device: PyTorch device
    
    Returns:
        QVector instance
    """
    _ensure_initialized()
    return QVector(n_qubits, batch_size, device)
