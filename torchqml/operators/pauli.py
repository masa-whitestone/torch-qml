"""Pauli operators and Hamiltonian construction"""

import torch
from typing import Union, List, Tuple
from numbers import Number


class PauliOperator:
    """
    Class representing a Pauli operator
    
    Internal representation: List[tuple(coeff, [(pauli, qubit), ...])]
    Example: 0.5 * X(0) @ Z(1) -> [(0.5, [("X", 0), ("Z", 1)])]
    """
    
    def __init__(self, terms: List[Tuple[float, List[Tuple[str, int]]]] = None):
        """
        Args:
            terms: [(coefficient, [(pauli_type, qubit_index), ...]), ...]
        """
        self.terms = terms if terms is not None else []
    
    @classmethod
    def single(cls, pauli_type: str, qubit: int) -> 'PauliOperator':
        """Create single Pauli operator"""
        return cls([(1.0, [(pauli_type, qubit)])])
    
    def __matmul__(self, other: 'PauliOperator') -> 'PauliOperator':
        """Tensor product (X(0) @ Z(1))"""
        if not isinstance(other, PauliOperator):
            raise TypeError(f"Cannot compute tensor product with {type(other)}")
        
        new_terms = []
        for coeff1, paulis1 in self.terms:
            for coeff2, paulis2 in other.terms:
                new_coeff = coeff1 * coeff2
                new_paulis = paulis1 + paulis2
                new_terms.append((new_coeff, new_paulis))
        
        return PauliOperator(new_terms)
    
    def __mul__(self, scalar: Union[Number, torch.Tensor]) -> 'PauliOperator':
        """Scalar multiplication"""
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.item()
        new_terms = [(coeff * scalar, paulis) for coeff, paulis in self.terms]
        return PauliOperator(new_terms)
    
    def __rmul__(self, scalar: Union[Number, torch.Tensor]) -> 'PauliOperator':
        """Right scalar multiplication"""
        return self.__mul__(scalar)
    
    def __add__(self, other: Union['PauliOperator', Number]) -> 'PauliOperator':
        """Addition"""
        if isinstance(other, (int, float)):
            # Scalar addition = Add Identity term (conceptually, usually to Hamiltonian)
            # Representing scalar as Identity on NO qubits is tricky in this representation
            # but usually fine for Hamiltonians.
            new_terms = self.terms + [(other, [])]
            return PauliOperator(new_terms)
        elif isinstance(other, PauliOperator):
            new_terms = self.terms + other.terms
            return PauliOperator(new_terms)
        else:
            raise TypeError(f"Cannot add PauliOperator with {type(other)}")
    
    def __radd__(self, other: Union['PauliOperator', Number]) -> 'PauliOperator':
        """Right addition"""
        return self.__add__(other)
    
    def __sub__(self, other: Union['PauliOperator', Number]) -> 'PauliOperator':
        """Subtraction"""
        if isinstance(other, (int, float)):
            return self + (-other)
        elif isinstance(other, PauliOperator):
            return self + (-1.0 * other)
        else:
            raise TypeError(f"Cannot subtract {type(other)} from PauliOperator")
    
    def __rsub__(self, other: Union['PauliOperator', Number]) -> 'PauliOperator':
        """Right subtraction"""
        return (-1.0 * self) + other
    
    def __neg__(self) -> 'PauliOperator':
        """Negation"""
        return -1.0 * self
    
    def to_cudaq_spin_op(self):
        """Convert to CUDA-Q spin_op"""
        try:
            import cudaq
            from cudaq import spin
        except ImportError:
            raise ImportError("CUDA-Q is required for converting to spin_op")
        
        result = None
        
        for coeff, paulis in self.terms:
            if len(paulis) == 0:
                # Identity term (scalar)
                # CUDA-Q spin op doesn't support pure scalar? 
                # Usually we multiply by I(0) or similar if it's part of Hamiltonian.
                # Or just spin.i(0) * coeff?
                # Actually cudaq.spin.i(0) is Identity on qubit 0.
                # If term is constant offset, valid approach is adding it to result.
                # But `spin` objects overloading `+` with scalars might work.
                # Let's assume we multiply by I(0) if no qubits specified, 
                # or just add scalar if supported.
                # Safest bet for generic scalar is often avoided or attached to I(0).
                # But here, let's try assuming at least one qubit exists or use I on qubit 0 as anchor if needed.
                # For now, let's map scalar to scalar if result exists, else scalar * I(0).
                term = coeff
                # If we have mixed scalar and operator, maybe problem.
                pass
            else:
                term = coeff
                for pauli_type, qubit in paulis:
                    if pauli_type == "X":
                        term = term * spin.x(qubit)
                    elif pauli_type == "Y":
                        term = term * spin.y(qubit)
                    elif pauli_type == "Z":
                        term = term * spin.z(qubit)
                    elif pauli_type == "I":
                        term = term * spin.i(qubit)
            
            if result is None:
                result = term
            else:
                result = result + term
        
        return result
    
    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        
        parts = []
        for coeff, paulis in self.terms:
            if len(paulis) == 0:
                parts.append(f"{coeff}")
            else:
                pauli_str = " @ ".join(f"{p}({q})" for p, q in paulis)
                if coeff == 1.0:
                    parts.append(pauli_str)
                elif coeff == -1.0:
                    parts.append(f"-{pauli_str}")
                else:
                    parts.append(f"{coeff} * {pauli_str}")
        
        return " + ".join(parts)


# Factory functions
def X(qubit: int) -> PauliOperator:
    """Pauli-X operator"""
    return PauliOperator.single("X", qubit)


def Y(qubit: int) -> PauliOperator:
    """Pauli-Y operator"""
    return PauliOperator.single("Y", qubit)


def Z(qubit: int) -> PauliOperator:
    """Pauli-Z operator"""
    return PauliOperator.single("Z", qubit)


def I(qubit: int) -> PauliOperator:
    """Identity operator"""
    return PauliOperator.single("I", qubit)
