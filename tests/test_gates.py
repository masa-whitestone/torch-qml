"""
ゲートの単体テスト
"""

import pytest
import torch
import numpy as np
import cupy as cp
from torchqml.backend import StateVector, GateType

class TestGates:
    def setup_method(self):
        self.n_qubits = 1
        self.sv = StateVector(self.n_qubits)
    
    def test_gate_matrices(self):
        """ゲート行列の検証"""
        # Pauli-X
        mat_x = self.sv.get_gate_matrix(GateType.X)
        expected_x = cp.array([[0, 1], [1, 0]], dtype=cp.complex64)
        cp.testing.assert_array_almost_equal(mat_x, expected_x)
        
        # Pauli-Z
        mat_z = self.sv.get_gate_matrix(GateType.Z)
        expected_z = cp.array([[1, 0], [0, -1]], dtype=cp.complex64)
        cp.testing.assert_array_almost_equal(mat_z, expected_z)
        
        # Hadamard
        mat_h = self.sv.get_gate_matrix(GateType.H)
        expected_h = cp.array([[1, 1], [1, -1]], dtype=cp.complex64) / cp.sqrt(2)
        cp.testing.assert_array_almost_equal(mat_h, expected_h)

    def test_parametric_gates(self):
        """パラメトリックゲートの検証"""
        theta = np.pi / 2
        
        # RX(pi/2)
        mat_rx = self.sv.get_gate_matrix(GateType.RX, theta)
        expected_rx = cp.array([
            [cp.cos(theta/2), -1j*cp.sin(theta/2)],
            [-1j*cp.sin(theta/2), cp.cos(theta/2)]
        ], dtype=cp.complex64)
        cp.testing.assert_array_almost_equal(mat_rx, expected_rx)
        
        # RY(pi/2)
        mat_ry = self.sv.get_gate_matrix(GateType.RY, theta)
        expected_ry = cp.array([
            [cp.cos(theta/2), -cp.sin(theta/2)],
            [cp.sin(theta/2), cp.cos(theta/2)]
        ], dtype=cp.complex64)
        cp.testing.assert_array_almost_equal(mat_ry, expected_ry)
        
    def test_gate_application(self):
        """ゲート適用の検証"""
        # |0> -> X -> |1>
        self.sv.apply_gate(self.sv.get_gate_matrix(GateType.X), [0])
        state = self.sv.get_state()
        expected = cp.array([0, 1], dtype=cp.complex64)
        cp.testing.assert_array_almost_equal(state, expected)
        
        # |1> -> H -> |-> = (|0> - |1>)/sqrt(2)
        self.sv.apply_gate(self.sv.get_gate_matrix(GateType.H), [0])
        state = self.sv.get_state()
        expected = cp.array([1, -1], dtype=cp.complex64) / cp.sqrt(2)
        cp.testing.assert_array_almost_equal(state, expected)

    def test_gate_derivatives(self):
        """ゲート微分の検証"""
        theta = np.pi / 3
        
        # dRY/dtheta
        # RY = [[c, -s], [s, c]]
        # dRY = [[-s/2, -c/2], [c/2, -s/2]]
        mat_dry = self.sv.get_gate_derivative(GateType.RY, theta)
        
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        expected = cp.array([
            [-s/2, -c/2],
            [c/2, -s/2]
        ], dtype=cp.complex64)
        
        cp.testing.assert_array_almost_equal(mat_dry, expected)
