"""
量子回路

ゲートを追加して回路を構築するためのクラス
"""

from typing import List, Optional, Union
from .backend import GateOp, GateType
from .operators import PauliOperator


class QuantumCircuit:
    """
    量子回路
    
    Example:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ry(0, param_index=0)
        qc.cx(0, 1)
        
        params = torch.tensor([[0.5]], requires_grad=True)
        exp_val = qc.expectation(params, tq.Z(0))
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self._operations: List[GateOp] = []
        self._n_params = 0
    
    @property
    def n_params(self) -> int:
        """パラメータ数"""
        return self._n_params
    
    @property
    def operations(self) -> List[GateOp]:
        """操作リスト"""
        return self._operations.copy()
    
    def _add_op(self, op: GateOp):
        """操作を追加"""
        self._operations.append(op)
    
    # ==================== 非パラメトリックゲート ====================
    
    def h(self, target: int) -> 'QuantumCircuit':
        """Hadamardゲート"""
        self._add_op(GateOp(GateType.H, [target]))
        return self
    
    def x(self, target: int) -> 'QuantumCircuit':
        """Pauli-Xゲート"""
        self._add_op(GateOp(GateType.X, [target]))
        return self
    
    def y(self, target: int) -> 'QuantumCircuit':
        """Pauli-Yゲート"""
        self._add_op(GateOp(GateType.Y, [target]))
        return self
    
    def z(self, target: int) -> 'QuantumCircuit':
        """Pauli-Zゲート"""
        self._add_op(GateOp(GateType.Z, [target]))
        return self
    
    def s(self, target: int) -> 'QuantumCircuit':
        """Sゲート"""
        self._add_op(GateOp(GateType.S, [target]))
        return self
    
    def t(self, target: int) -> 'QuantumCircuit':
        """Tゲート"""
        self._add_op(GateOp(GateType.T, [target]))
        return self
    
    # ==================== パラメトリックゲート ====================
    
    def rx(self, target: int, param_index: Optional[int] = None) -> 'QuantumCircuit':
        """RXゲート"""
        if param_index is None:
            param_index = self._n_params
            self._n_params += 1
        elif param_index >= self._n_params:
            self._n_params = param_index + 1
        self._add_op(GateOp(GateType.RX, [target], param_index=param_index))
        return self
    
    def ry(self, target: int, param_index: Optional[int] = None) -> 'QuantumCircuit':
        """RYゲート"""
        if param_index is None:
            param_index = self._n_params
            self._n_params += 1
        elif param_index >= self._n_params:
            self._n_params = param_index + 1
        self._add_op(GateOp(GateType.RY, [target], param_index=param_index))
        return self
    
    def rz(self, target: int, param_index: Optional[int] = None) -> 'QuantumCircuit':
        """RZゲート"""
        if param_index is None:
            param_index = self._n_params
            self._n_params += 1
        elif param_index >= self._n_params:
            self._n_params = param_index + 1
        self._add_op(GateOp(GateType.RZ, [target], param_index=param_index))
        return self
    
    # ==================== 2qubitゲート ====================
    
    def cx(self, control: int, target: int) -> 'QuantumCircuit':
        """CNOTゲート"""
        self._add_op(GateOp(GateType.CNOT, [target], [control]))
        return self
    
    def cnot(self, control: int, target: int) -> 'QuantumCircuit':
        """CNOTゲート（エイリアス）"""
        return self.cx(control, target)
    
    def cz(self, control: int, target: int) -> 'QuantumCircuit':
        """CZゲート"""
        self._add_op(GateOp(GateType.CZ, [target], [control]))
        return self
    
    def swap(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        """SWAPゲート"""
        self._add_op(GateOp(GateType.SWAP, [qubit1, qubit2]))
        return self
    
    # ==================== 実行 ====================
    
    def expectation(
        self,
        params: 'torch.Tensor',
        observable: PauliOperator
    ) -> 'torch.Tensor':
        """
        期待値を計算
        
        Args:
            params: パラメータ [batch_size, n_params] または [n_params]
            observable: 観測量
        
        Returns:
            期待値 [batch_size]
        """
        from .autograd import QuantumExpectation
        import torch
        
        # バッチ次元を追加
        if params.dim() == 1:
            params = params.unsqueeze(0)
        
        return QuantumExpectation.apply(
            params,
            self._operations,
            observable.to_observable(),
            self.n_qubits
        )
    
    def sample(self, params: 'torch.Tensor', n_shots: int = 1000) -> List[dict]:
        """
        サンプリング
        
        Args:
            params: パラメータ
            n_shots: ショット数
        
        Returns:
            カウント辞書のリスト
        """
        from .backend import StateVector, AdjointDifferentiator
        import torch
        
        if params.dim() == 1:
            params = params.unsqueeze(0)
        
        params_np = params.detach().cpu().numpy()
        batch_size = params_np.shape[0]
        
        results = []
        for b in range(batch_size):
            sv = StateVector(self.n_qubits)
            
            for op in self._operations:
                if op.param_index is not None:
                    param_val = float(params_np[b, op.param_index])
                    matrix = sv.get_gate_matrix(op.gate_type, param_val)
                else:
                    matrix = sv.get_gate_matrix(op.gate_type)
                
                if op.gate_type == GateType.CNOT:
                    sv.apply_cnot(op.controls[0], op.targets[0])
                elif op.gate_type == GateType.CZ:
                    sv.apply_cz(op.controls[0], op.targets[0])
                elif op.gate_type == GateType.SWAP:
                    sv.apply_swap(op.targets[0], op.targets[1])
                else:
                    sv.apply_gate(matrix, op.targets, op.controls)
            
            results.append(sv.sample(n_shots))
        
        return results
    
    def __repr__(self) -> str:
        lines = [f"QuantumCircuit({self.n_qubits} qubits, {self.n_params} params)"]
        for i, op in enumerate(self._operations):
            lines.append(f"  {i}: {op.gate_type.value} on {op.targets} "
                        f"(ctrl={op.controls}, param={op.param_index})")
        return "\n".join(lines)
