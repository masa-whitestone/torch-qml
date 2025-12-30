
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# C++バックエンドのインポートを試行
try:
    from torchqml._C import StateVector, GateOp, GateType, adjoint_differentiate
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

@pytest.mark.skipif(not HAS_CPP, reason="C++ backend not available")
class TestCppBackend:
    def test_state_vector_initialization(self):
        n_qubits = 2
        sv = StateVector(n_qubits)
        assert sv is not None
        
        # 初期状態は |0...0> なので State[0] = 1+0j
        # get_amplitudes() のようなメソッドがあればテストしやすいが、
        # 現状は C++ 側で計算して期待値を返す設計。
        # ここでは expectation_z を使って確認
        
        # <00| Z0 |00> = 1
        exp_z0 = sv.expectation_z(0)
        assert np.isclose(exp_z0, 1.0)
        
        # <00| Z1 |00> = 1
        exp_z1 = sv.expectation_z(1)
        assert np.isclose(exp_z1, 1.0)

    def test_gate_application(self):
        n_qubits = 1
        sv = StateVector(n_qubits)
        
        # Xゲート: |0> -> |1>
        # <1| Z |1> = -1
        sv.apply_x(0)
        assert np.isclose(sv.expectation_z(0), -1.0)
        
        # Hゲート: |1> -> |-> = (|0> - |1>)/sqrt(2)
        # <| Z |> = 0
        sv.apply_h(0)
        assert np.isclose(sv.expectation_z(0), 0.0, atol=1e-5)

    def test_rx_gate(self):
        n_qubits = 1
        sv = StateVector(n_qubits)
        
        theta = np.pi
        # RX(pi) |0> = -i |1>
        # <psi| Z |psi> = -1
        sv.apply_rx(0, theta)
        assert np.isclose(sv.expectation_z(0), -1.0, atol=1e-5)

    def test_cnot(self):
        n_qubits = 2
        sv = StateVector(n_qubits)
        
        # |00> -> X0 -> |10> -> CNOT(0, 1) -> |11>
        sv.apply_x(0)
        sv.apply_cnot(0, 1) # control: 0, target: 1
        
        assert np.isclose(sv.expectation_z(0), -1.0) # |1>
        assert np.isclose(sv.expectation_z(1), -1.0) # |1>

    def test_adjoint_differentiation_simple(self):
        # RX(theta) |0>
        # Expectation Z
        # E = <0| RX(-t) Z RX(t) |0>
        # E = cos(theta)
        # dE/dt = -sin(theta)
        
        n_qubits = 1
        theta = 1.5
        
        # Make params 2D (batch_size=1, n_params=1)
        params = np.array([[theta]], dtype=np.float32)
        
        # GateOp作成
        op = GateOp()
        op.gate_type = GateType.RX
        op.targets = [0]
        op.param_index = 0
        
        operations = [op]
        
        # Observable: Z0
        # coeff=1.0, paulis=[("z", 0)]
        from torchqml._C import PauliTerm
        term = PauliTerm()
        term.coefficient = 1.0
        term.paulis = [("Z", 0)] # Capital Z usually
        
        observable = [term]
        
        exp_val, grads = adjoint_differentiate(params, operations, observable, n_qubits)
        
        expected_exp = np.cos(theta)
        expected_grad = -np.sin(theta)
        
        # exp_val: (batch_size,) -> (1,)
        # grads: (batch_size, n_params) -> (1, 1)
        assert np.isclose(exp_val[0], expected_exp, atol=1e-4)
        assert np.isclose(grads[0, 0], expected_grad, atol=1e-4)
