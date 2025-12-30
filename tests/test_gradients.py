"""
勾配計算の検証
"""

import pytest
import torch
import numpy as np
import torchqml as tq

def test_ry_gradient():
    """RYゲートの勾配検証"""
    qc = tq.QuantumCircuit(1)
    qc.ry(0, param_index=0)
    
    # params = pi/2
    # <Z> = <0|RY(-pi/2) Z RY(pi/2)|0>
    # RY(theta)|0> = cos(t/2)|0> + sin(t/2)|1>
    # <psi|Z|psi> = cos^2(t/2) - sin^2(t/2) = cos(theta)
    # d/dtheta cos(theta) = -sin(theta)
    # at theta=pi/2, grad = -1
    
    params = torch.tensor([[np.pi / 2]], requires_grad=True)
    exp_val = qc.expectation(params, tq.Z(0))
    exp_val.backward()
    
    expected_grad = -1.0
    assert torch.allclose(params.grad, torch.tensor([[expected_grad]]), atol=1e-5)

def test_rx_gradient():
    """RXゲートの勾配検証"""
    qc = tq.QuantumCircuit(1)
    qc.rx(0, param_index=0)
    
    # RX(theta)|0> = cos(t/2)|0> - i sin(t/2)|1>
    # <Z> = cos^2(t/2) - sin^2(t/2) = cos(theta) (wait, | -i sin|^2 = sin^2)
    # Yes, <Z> = cos(theta)
    
    params = torch.tensor([[np.pi / 2]], requires_grad=True)
    exp_val = qc.expectation(params, tq.Z(0))
    exp_val.backward()
    
    expected_grad = -1.0
    assert torch.allclose(params.grad, torch.tensor([[expected_grad]]), atol=1e-5)

def test_parameter_shift_comparison():
    """Parameter Shiftとの比較（マルチパラメータ）"""
    qc = tq.QuantumCircuit(2)
    qc.h(0)
    qc.ry(0, param_index=0)
    qc.rx(1, param_index=1)
    qc.cx(0, 1)
    
    params = torch.tensor([[0.3, 0.7]], requires_grad=True)
    
    # Adjoint Diff
    exp_val = qc.expectation(params, tq.Z(0) @ tq.Z(1))
    exp_val.backward()
    adjoint_grad = params.grad.clone()
    
    # Parameter Shift
    params.grad = None
    shift = np.pi / 2
    ps_grad = torch.zeros_like(params)
    
    for i in range(2):
        p_plus = params.detach().clone()
        p_plus[0, i] += shift
        p_minus = params.detach().clone()
        p_minus[0, i] -= shift
        
        with torch.no_grad():
            exp_plus = qc.expectation(p_plus, tq.Z(0) @ tq.Z(1))
            exp_minus = qc.expectation(p_minus, tq.Z(0) @ tq.Z(1))
        
        ps_grad[0, i] = (exp_plus - exp_minus) / 2
    
    assert torch.allclose(adjoint_grad, ps_grad, atol=1e-4)

def test_batch_processing():
    """バッチ処理の検証"""
    qc = tq.QuantumCircuit(1)
    qc.ry(0, param_index=0)
    
    # Batch size = 3
    params = torch.tensor([
        [0.0],
        [np.pi/2],
        [np.pi]
    ], requires_grad=True)
    
    # Expected <Z>: [cos(0), cos(pi/2), cos(pi)] = [1, 0, -1]
    exp_val = qc.expectation(params, tq.Z(0))
    
    expected_val = torch.tensor([1.0, 0.0, -1.0])
    assert torch.allclose(exp_val, expected_val, atol=1e-5)
    
    # Check gradients
    # d/dtheta <Z> = -sin(theta)
    # [-sin(0), -sin(pi/2), -sin(pi)] = [0, -1, 0]
    exp_val.sum().backward()
    
    expected_grad = torch.tensor([
        [0.0],
        [-1.0],
        [0.0]
    ])
    assert torch.allclose(params.grad, expected_grad, atol=1e-5)
