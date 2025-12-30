"""
関数型API

PennyLane風の関数型インターフェースを提供
"""

from typing import Union, Optional
from .circuit import QuantumCircuit


# ゲート関数（回路に追加）

def h(qc: QuantumCircuit, target: int) -> QuantumCircuit:
    """Hadamard"""
    return qc.h(target)

def x(qc: QuantumCircuit, target: int) -> QuantumCircuit:
    """Pauli-X"""
    return qc.x(target)

def y(qc: QuantumCircuit, target: int) -> QuantumCircuit:
    """Pauli-Y"""
    return qc.y(target)

def z(qc: QuantumCircuit, target: int) -> QuantumCircuit:
    """Pauli-Z"""
    return qc.z(target)

def rx(qc: QuantumCircuit, target: int, param_index: Optional[int] = None) -> QuantumCircuit:
    """RX rotation"""
    return qc.rx(target, param_index)

def ry(qc: QuantumCircuit, target: int, param_index: Optional[int] = None) -> QuantumCircuit:
    """RY rotation"""
    return qc.ry(target, param_index)

def rz(qc: QuantumCircuit, target: int, param_index: Optional[int] = None) -> QuantumCircuit:
    """RZ rotation"""
    return qc.rz(target, param_index)

def cx(qc: QuantumCircuit, control: int, target: int) -> QuantumCircuit:
    """CNOT"""
    return qc.cx(control, target)

def cnot(qc: QuantumCircuit, control: int, target: int) -> QuantumCircuit:
    """CNOT (alias)"""
    return qc.cx(control, target)

def cz(qc: QuantumCircuit, control: int, target: int) -> QuantumCircuit:
    """CZ"""
    return qc.cz(control, target)

def swap(qc: QuantumCircuit, qubit1: int, qubit2: int) -> QuantumCircuit:
    """SWAP"""
    return qc.swap(qubit1, qubit2)
