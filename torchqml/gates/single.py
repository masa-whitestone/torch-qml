"""Single qubit gates"""

import torch
from typing import Union, Optional, List
from ..core.qvector import QVector, QubitRef, QubitRefSlice


def _normalize_target(target: Union[QVector, QubitRef, QubitRefSlice]) -> tuple:
    """Normalize target to (qvector, indices)"""
    if isinstance(target, QVector):
        # Apply to all qubits
        return target, list(range(target.n_qubits))
    elif isinstance(target, QubitRef):
        return target.qvector, [target.index]
    elif isinstance(target, QubitRefSlice):
        return target.qvector, target.indices
    else:
        raise TypeError(f"Invalid target type: {type(target)}")


def _add_gate(
    target: Union[QVector, QubitRef, QubitRefSlice],
    gate_name: str,
    params: Optional[torch.Tensor] = None,
    ctrl_indices: Optional[list] = None,
    adjoint: bool = False
):
    """Record gate operation"""
    qvector, target_indices = _normalize_target(target)
    
    op = {
        "gate": gate_name,
        "targets": target_indices,
        "params": params,
        "controls": ctrl_indices,
        "adjoint": adjoint,
    }
    qvector._add_operation(op)


# ================== Non-parametric gates ==================

class GateH:
    """Hadamard gate"""
    
    def __call__(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        _add_gate(target, "h")
    
    def ctrl(
        self, 
        control: Union[QubitRef, List[QubitRef], QubitRefSlice], 
        target: QubitRef
    ):
        """Controlled-Hadamard"""
        if isinstance(control, QubitRef):
            ctrl_indices = [control.index]
        elif isinstance(control, QubitRefSlice):
            ctrl_indices = control.indices
        else:
            ctrl_indices = [c.index for c in control]
        _add_gate(target, "h", ctrl_indices=ctrl_indices)
    
    def adj(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        """Hadamard adjoint (H^dagger = H)"""
        _add_gate(target, "h", adjoint=True)


class GateX:
    """Pauli-X gate"""
    
    def __call__(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        _add_gate(target, "x")
    
    def ctrl(
        self, 
        control: Union[QubitRef, List[QubitRef], QubitRefSlice], 
        target: QubitRef
    ):
        """CNOT / Toffoli"""
        if isinstance(control, QubitRef):
            ctrl_indices = [control.index]
        elif isinstance(control, QubitRefSlice):
            ctrl_indices = control.indices
        else:
            ctrl_indices = [c.index for c in control]
        _add_gate(target, "x", ctrl_indices=ctrl_indices)
    
    def adj(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        """X^dagger = X"""
        _add_gate(target, "x", adjoint=True)


class GateY:
    """Pauli-Y gate"""
    
    def __call__(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        _add_gate(target, "y")
    
    def ctrl(self, control: Union[QubitRef, List[QubitRef], QubitRefSlice], target: QubitRef):
        if isinstance(control, QubitRef):
            ctrl_indices = [control.index]
        elif isinstance(control, QubitRefSlice):
            ctrl_indices = control.indices
        else:
            ctrl_indices = [c.index for c in control]
        _add_gate(target, "y", ctrl_indices=ctrl_indices)
    
    def adj(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        _add_gate(target, "y", adjoint=True)


class GateZ:
    """Pauli-Z gate"""
    
    def __call__(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        _add_gate(target, "z")
    
    def ctrl(self, control: Union[QubitRef, List[QubitRef], QubitRefSlice], target: QubitRef):
        if isinstance(control, QubitRef):
            ctrl_indices = [control.index]
        elif isinstance(control, QubitRefSlice):
            ctrl_indices = control.indices
        else:
            ctrl_indices = [c.index for c in control]
        _add_gate(target, "z", ctrl_indices=ctrl_indices)
    
    def adj(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        _add_gate(target, "z", adjoint=True)


class GateS:
    """S gate (pi/2 Z rotation)"""
    
    def __call__(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        _add_gate(target, "s")
    
    def ctrl(self, control: Union[QubitRef, List[QubitRef], QubitRefSlice], target: QubitRef):
        if isinstance(control, QubitRef):
            ctrl_indices = [control.index]
        elif isinstance(control, QubitRefSlice):
            ctrl_indices = control.indices
        else:
            ctrl_indices = [c.index for c in control]
        _add_gate(target, "s", ctrl_indices=ctrl_indices)
    
    def adj(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        """S^dagger (Sdg)"""
        _add_gate(target, "s", adjoint=True)


class GateT:
    """T gate (pi/4 Z rotation)"""
    
    def __call__(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        _add_gate(target, "t")
    
    def ctrl(self, control: Union[QubitRef, List[QubitRef], QubitRefSlice], target: QubitRef):
        if isinstance(control, QubitRef):
            ctrl_indices = [control.index]
        elif isinstance(control, QubitRefSlice):
            ctrl_indices = control.indices
        else:
            ctrl_indices = [c.index for c in control]
        _add_gate(target, "t", ctrl_indices=ctrl_indices)
    
    def adj(self, target: Union[QVector, QubitRef, QubitRefSlice]):
        """T^dagger (Tdg)"""
        _add_gate(target, "t", adjoint=True)


# Singleton instances
h = GateH()
x = GateX()
y = GateY()
z = GateZ()
s = GateS()
t = GateT()
