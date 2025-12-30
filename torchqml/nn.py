"""
PyTorch nn.Module統合

量子層をnn.Moduleとして使用するためのクラス
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable
import math

from .circuit import QuantumCircuit
from .operators import PauliOperator, Z


class QuantumLayer(nn.Module):
    """
    量子回路層
    
    Example:
        class MyCircuit(tq.QuantumLayer):
            def build_circuit(self, qc: QuantumCircuit):
                qc.h(0)
                qc.ry(0, param_index=0)
                qc.cx(0, 1)
            
            def observable(self):
                return tq.Z(0)
        
        layer = MyCircuit(n_qubits=2, n_params=1)
        output = layer()  # [1] expectation
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_params: int,
        observable: Optional[PauliOperator] = None
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self._n_params = n_params
        self._observable = observable
        
        # 学習可能パラメータ
        self.params = nn.Parameter(torch.empty(n_params))
        self._init_params()
        
        # 回路をビルド
        self._circuit = QuantumCircuit(n_qubits)
        self.build_circuit(self._circuit)
    
    def _init_params(self):
        """パラメータ初期化"""
        nn.init.uniform_(self.params, -math.pi, math.pi)
    
    def build_circuit(self, qc: QuantumCircuit):
        """
        回路を構築（サブクラスでオーバーライド）
        """
        raise NotImplementedError
    
    def get_observable(self) -> PauliOperator:
        """観測量を取得"""
        if self._observable is not None:
            return self._observable
        return self.observable()
    
    def observable(self) -> PauliOperator:
        """観測量を定義（サブクラスでオーバーライド可能）"""
        return Z(0)
    
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            batch_size: バッチサイズ
        
        Returns:
            期待値 [batch_size]
        """
        # パラメータをバッチに展開
        params = self.params.unsqueeze(0).expand(batch_size, -1)
        
        return self._circuit.expectation(params, self.get_observable())


class HybridQuantumLayer(nn.Module):
    """
    ハイブリッド量子層（古典入力を受け取る）
    
    Example:
        class MyHybridCircuit(tq.HybridQuantumLayer):
            def build_circuit(self, qc: QuantumCircuit, n_classical: int, n_quantum: int):
                # classical inputs: param_index 0 ~ n_classical-1
                # quantum params: param_index n_classical ~ n_classical+n_quantum-1
                qc.ry(0, param_index=0)  # classical input
                qc.rx(0, param_index=n_classical)  # quantum param
                qc.cx(0, 1)
        
        layer = MyHybridCircuit(n_qubits=2, n_classical=2, n_quantum=2)
        x = torch.randn(32, 2)
        output = layer(x)  # [32] expectation
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_classical: int,
        n_quantum: int = 0,
        observable: Optional[PauliOperator] = None
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classical = n_classical
        self.n_quantum = n_quantum
        self._observable = observable
        
        # 量子パラメータ
        if n_quantum > 0:
            self.quantum_params = nn.Parameter(torch.empty(n_quantum))
            nn.init.uniform_(self.quantum_params, -math.pi, math.pi)
        else:
            self.register_parameter('quantum_params', None)
        
        # 回路をビルド
        self._circuit = QuantumCircuit(n_qubits)
        self.build_circuit(self._circuit, n_classical, n_quantum)
    
    def build_circuit(self, qc: QuantumCircuit, n_classical: int, n_quantum: int):
        """回路を構築（サブクラスでオーバーライド）"""
        raise NotImplementedError
    
    def get_observable(self) -> PauliOperator:
        """観測量を取得"""
        if self._observable is not None:
            return self._observable
        return self.observable()
    
    def observable(self) -> PauliOperator:
        """観測量を定義"""
        return Z(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 古典入力 [batch_size, n_classical]
        
        Returns:
            期待値 [batch_size]
        """
        batch_size = x.size(0)
        
        # 古典入力 + 量子パラメータを結合
        if self.quantum_params is not None:
            quantum_expanded = self.quantum_params.unsqueeze(0).expand(batch_size, -1)
            params = torch.cat([x, quantum_expanded], dim=1)
        else:
            params = x
        
        return self._circuit.expectation(params, self.get_observable())


class StronglyEntanglingLayer(QuantumLayer):
    """
    Strongly Entangling Layer
    
    PennyLaneのStronglyEntanglingLayersに相当
    各層: 全qubitにRY-RZ回転 → 環状CNOT
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        observable: Optional[PauliOperator] = None
    ):
        self.n_layers = n_layers
        n_params = n_qubits * n_layers * 2  # RY + RZ per qubit per layer
        super().__init__(n_qubits, n_params, observable)
    
    def build_circuit(self, qc: QuantumCircuit):
        """Strongly Entangling回路を構築"""
        param_idx = 0
        
        for layer in range(self.n_layers):
            # 単一qubit回転
            for q in range(self.n_qubits):
                qc.ry(q, param_index=param_idx)
                param_idx += 1
                qc.rz(q, param_index=param_idx)
                param_idx += 1
            
            # 環状CNOT
            for q in range(self.n_qubits):
                qc.cx(q, (q + 1) % self.n_qubits)
