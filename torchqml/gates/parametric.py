"""Parametric gates"""

import torch
from typing import Union, Optional, List
from ..core.qvector import QVector, QubitRef, QubitRefSlice


def _normalize_params(
    params: Union[float, torch.Tensor], 
    batch_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Normalize parameters to batch tensor
    
    - Scalar -> Expand to [batch_size]
    - [batch_size] Tensor -> As is
    """
    if isinstance(params, (int, float)):
        return torch.full((batch_size,), params, device=device, dtype=torch.float32)
    elif isinstance(params, torch.Tensor):
        if params.dim() == 0:
            # Scalar tensor
            return params.expand(batch_size).to(device)
        elif params.shape[0] == batch_size:
            return params.to(device)
        elif params.shape[0] == 1:
            return params.expand(batch_size).to(device)
        else:
            raise ValueError(
                f"Parameter batch size {params.shape[0]} doesn't match "
                f"qvector batch size {batch_size}"
            )
    else:
        raise TypeError(f"Invalid parameter type: {type(params)}")


def _add_parametric_gate(
    target: Union[QVector, QubitRef, QubitRefSlice],
    gate_name: str,
    params: Union[float, torch.Tensor],
    ctrl_indices: Optional[list] = None,
    adjoint: bool = False
):
    """Record parametric gate operation"""
    if isinstance(target, QVector):
        qvector = target
        target_indices = list(range(target.n_qubits))
    elif isinstance(target, QubitRef):
        qvector = target.qvector
        target_indices = [target.index]
    elif isinstance(target, QubitRefSlice):
        qvector = target.qvector
        target_indices = target.indices
    else:
        raise TypeError(f"Invalid target type: {type(target)}")
    
    # Normalize parameters
    normalized_params = _normalize_params(params, qvector.batch_size, qvector.device)
    
    op = {
        "gate": gate_name,
        "targets": target_indices,
        "params": normalized_params,
        "controls": ctrl_indices,
        "adjoint": adjoint,
    }
    qvector._add_operation(op)


class GateRX:
    """RX rotation gate"""
    
    def __call__(
        self, 
        theta: Union[float, torch.Tensor], 
        target: Union[QVector, QubitRef, QubitRefSlice]
    ):
        _add_parametric_gate(target, "rx", theta)
    
    def ctrl(
        self,
        theta: Union[float, torch.Tensor],
        control: Union[QubitRef, List[QubitRef], QubitRefSlice],
        target: QubitRef
    ):
        if isinstance(control, QubitRef):
            ctrl_indices = [control.index]
        elif isinstance(control, QubitRefSlice):
            ctrl_indices = control.indices
        else:
            ctrl_indices = [c.index for c in control]
        _add_parametric_gate(target, "rx", theta, ctrl_indices=ctrl_indices)
    
    def adj(
        self,
        theta: Union[float, torch.Tensor],
        target: Union[QVector, QubitRef, QubitRefSlice]
    ):
        _add_parametric_gate(target, "rx", theta, adjoint=True)


class GateRY:
    """RY rotation gate"""
    
    def __call__(
        self, 
        theta: Union[float, torch.Tensor], 
        target: Union[QVector, QubitRef, QubitRefSlice]
    ):
        _add_parametric_gate(target, "ry", theta)
    
    def ctrl(
        self,
        theta: Union[float, torch.Tensor],
        control: Union[QubitRef, List[QubitRef], QubitRefSlice],
        target: QubitRef
    ):
        if isinstance(control, QubitRef):
            ctrl_indices = [control.index]
        elif isinstance(control, QubitRefSlice):
            ctrl_indices = control.indices
        else:
            ctrl_indices = [c.index for c in control]
        _add_parametric_gate(target, "ry", theta, ctrl_indices=ctrl_indices)
    
    def adj(
        self,
        theta: Union[float, torch.Tensor],
        target: Union[QVector, QubitRef, QubitRefSlice]
    ):
        _add_parametric_gate(target, "ry", theta, adjoint=True)


class GateRZ:
    """RZ rotation gate"""
    
    def __call__(
        self, 
        theta: Union[float, torch.Tensor], 
        target: Union[QVector, QubitRef, QubitRefSlice]
    ):
        _add_parametric_gate(target, "rz", theta)
    
    def ctrl(
        self,
        theta: Union[float, torch.Tensor],
        control: Union[QubitRef, List[QubitRef], QubitRefSlice],
        target: QubitRef
    ):
        if isinstance(control, QubitRef):
            ctrl_indices = [control.index]
        elif isinstance(control, QubitRefSlice):
            ctrl_indices = control.indices
        else:
            ctrl_indices = [c.index for c in control]
        _add_parametric_gate(target, "rz", theta, ctrl_indices=ctrl_indices)
    
    def adj(
        self,
        theta: Union[float, torch.Tensor],
        target: Union[QVector, QubitRef, QubitRefSlice]
    ):
        _add_parametric_gate(target, "rz", theta, adjoint=True)


class GateR1:
    """R1 rotation gate (phase rotation for |1>)"""
    
    def __call__(
        self, 
        theta: Union[float, torch.Tensor], 
        target: Union[QVector, QubitRef, QubitRefSlice]
    ):
        _add_parametric_gate(target, "r1", theta)
    
    def ctrl(
        self,
        theta: Union[float, torch.Tensor],
        control: Union[QubitRef, List[QubitRef], QubitRefSlice],
        target: QubitRef
    ):
        if isinstance(control, QubitRef):
            ctrl_indices = [control.index]
        elif isinstance(control, QubitRefSlice):
            ctrl_indices = control.indices
        else:
            ctrl_indices = [c.index for c in control]
        _add_parametric_gate(target, "r1", theta, ctrl_indices=ctrl_indices)
    
    def adj(
        self,
        theta: Union[float, torch.Tensor],
        target: Union[QVector, QubitRef, QubitRefSlice]
    ):
        _add_parametric_gate(target, "r1", theta, adjoint=True)


def u3(
    theta: Union[float, torch.Tensor],
    phi: Union[float, torch.Tensor],
    lam: Union[float, torch.Tensor],
    target: Union[QVector, QubitRef, QubitRefSlice]
):
    """
    U3 gate (3-parameter general single qubit gate)
    
    U3(theta, phi, lambda) = | cos(theta/2)           -exp(ilambda)sin(theta/2)      |
                             | exp(iphi)sin(theta/2)   exp(i(phi+lambda))cos(theta/2) |
    """
    if isinstance(target, QVector):
        qvector = target
        target_indices = list(range(target.n_qubits))
    elif isinstance(target, QubitRef):
        qvector = target.qvector
        target_indices = [target.index]
    else:
        qvector = target.qvector
        target_indices = target.indices
    
    # Normalize parameters
    theta_norm = _normalize_params(theta, qvector.batch_size, qvector.device)
    phi_norm = _normalize_params(phi, qvector.batch_size, qvector.device)
    lam_norm = _normalize_params(lam, qvector.batch_size, qvector.device)
    
    # Stack parameters [batch_size, 3]
    params = torch.stack([theta_norm, phi_norm, lam_norm], dim=1)
    
    op = {
        "gate": "u3",
        "targets": target_indices,
        "params": params,
        "controls": None,
        "adjoint": False,
    }
    qvector._add_operation(op)


# Singleton instances
rx = GateRX()
ry = GateRY()
rz = GateRZ()
r1 = GateR1()
