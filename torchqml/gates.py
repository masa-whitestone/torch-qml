"""
ゲート定義

単一qubitゲート、パラメトリックゲート、2qubitゲートを提供
"""

from typing import Union, List, Optional
from .backend import GateType, GateOp


class Gate:
    """ゲート基底クラス"""
    pass


# ==================== 非パラメトリックゲート ====================

class H(Gate):
    """Hadamardゲート"""
    
    def __init__(self, target: int, controls: List[int] = None):
        self.target = target
        self.controls = controls or []
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.H,
            targets=[self.target],
            controls=self.controls
        )


class X(Gate):
    """Pauli-Xゲート"""
    
    def __init__(self, target: int, controls: List[int] = None):
        self.target = target
        self.controls = controls or []
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.X,
            targets=[self.target],
            controls=self.controls
        )


class Y(Gate):
    """Pauli-Yゲート"""
    
    def __init__(self, target: int, controls: List[int] = None):
        self.target = target
        self.controls = controls or []
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.Y,
            targets=[self.target],
            controls=self.controls
        )


class Z(Gate):
    """Pauli-Zゲート"""
    
    def __init__(self, target: int, controls: List[int] = None):
        self.target = target
        self.controls = controls or []
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.Z,
            targets=[self.target],
            controls=self.controls
        )


class S(Gate):
    """Sゲート（π/2位相）"""
    
    def __init__(self, target: int):
        self.target = target
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.S,
            targets=[self.target]
        )


class T(Gate):
    """Tゲート（π/4位相）"""
    
    def __init__(self, target: int):
        self.target = target
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.T,
            targets=[self.target]
        )


# ==================== パラメトリックゲート ====================

class RX(Gate):
    """RX回転ゲート"""
    
    def __init__(self, target: int, param_index: int, controls: List[int] = None):
        self.target = target
        self.param_index = param_index
        self.controls = controls or []
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.RX,
            targets=[self.target],
            controls=self.controls,
            param_index=self.param_index
        )


class RY(Gate):
    """RY回転ゲート"""
    
    def __init__(self, target: int, param_index: int, controls: List[int] = None):
        self.target = target
        self.param_index = param_index
        self.controls = controls or []
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.RY,
            targets=[self.target],
            controls=self.controls,
            param_index=self.param_index
        )


class RZ(Gate):
    """RZ回転ゲート"""
    
    def __init__(self, target: int, param_index: int, controls: List[int] = None):
        self.target = target
        self.param_index = param_index
        self.controls = controls or []
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.RZ,
            targets=[self.target],
            controls=self.controls,
            param_index=self.param_index
        )


# ==================== 2qubitゲート ====================

class CNOT(Gate):
    """CNOTゲート"""
    
    def __init__(self, control: int, target: int):
        self.control = control
        self.target = target
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.CNOT,
            targets=[self.target],
            controls=[self.control]
        )


class CZ(Gate):
    """CZゲート"""
    
    def __init__(self, control: int, target: int):
        self.control = control
        self.target = target
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.CZ,
            targets=[self.target],
            controls=[self.control]
        )


class SWAP(Gate):
    """SWAPゲート"""
    
    def __init__(self, qubit1: int, qubit2: int):
        self.qubit1 = qubit1
        self.qubit2 = qubit2
    
    def to_op(self) -> GateOp:
        return GateOp(
            gate_type=GateType.SWAP,
            targets=[self.qubit1, self.qubit2]
        )


# エイリアス
CX = CNOT
