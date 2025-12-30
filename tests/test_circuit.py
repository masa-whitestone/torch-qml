"""
回路構築の検証
"""

import pytest
import torch
import torchqml as tq
from torchqml.backend import GateType

def test_circuit_construction():
    """回路構築APIのテスト"""
    qc = tq.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(1, param_index=0)
    
    assert qc.n_qubits == 2
    assert qc.n_params == 1
    assert len(qc.operations) == 3
    
    ops = qc.operations
    assert ops[0].gate_type == GateType.H
    assert ops[0].targets == [0]
    
    assert ops[1].gate_type == GateType.CNOT
    assert ops[1].targets == [1]
    assert ops[1].controls == [0]
    
    assert ops[2].gate_type == GateType.RY
    assert ops[2].targets == [1]
    assert ops[2].param_index == 0

def test_functional_api():
    """関数型APIのテスト"""
    qc = tq.QuantumCircuit(2)
    tq.functional.h(qc, 0)
    tq.functional.cx(qc, 0, 1)
    
    assert len(qc.operations) == 2
    assert qc.operations[0].gate_type == GateType.H
    assert qc.operations[1].gate_type == GateType.CNOT

def test_circuit_repr():
    """__repr__のテスト"""
    qc = tq.QuantumCircuit(2)
    qc.h(0)
    s = repr(qc)
    assert "QuantumCircuit(2 qubits" in s
    assert "h on [0]" in s
