"""
PyTorch autograd統合

torch.autograd.Functionを使用してAdjoint Differentiationを統合
"""

import torch
from torch.autograd import Function
from typing import List, Tuple, Any
import numpy as np

from .backend import AdjointDifferentiator, GateOp


class QuantumExpectation(Function):
    """
    量子期待値計算のためのautograd Function
    
    forward: 期待値を計算
    backward: Adjoint Differentiationで勾配を計算
    """
    
    @staticmethod
    def forward(
        ctx,
        params: torch.Tensor,                              # [batch_size, n_params]
        operations: List[GateOp],                          # 回路操作
        observable: List[Tuple[float, List[Tuple[str, int]]]],  # ハミルトニアン
        n_qubits: int
    ) -> torch.Tensor:
        """Forward pass"""
        
        device = params.device
        params_np = params.detach().cpu().numpy()
        
        # Adjoint Differentiatorを使用
        diff = AdjointDifferentiator(n_qubits)
        
        if params.requires_grad:
            # 勾配が必要な場合: forward + gradientを同時計算
            expectations, gradients = diff.forward_and_gradient(
                params_np, operations, observable
            )
            # 勾配を保存
            ctx.save_for_backward(torch.from_numpy(gradients).to(device))
        else:
            # 勾配不要の場合: forward only
            expectations = diff.forward_only(params_np, operations, observable)
        
        ctx.device = device
        
        return torch.from_numpy(expectations).float().to(device)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass"""
        
        gradients, = ctx.saved_tensors
        
        # grad_output: [batch_size]
        # gradients: [batch_size, n_params]
        # チェーンルール: grad_params = gradients * grad_output.unsqueeze(1)
        
        grad_params = gradients.float() * grad_output.unsqueeze(1)
        
        # operations, observable, n_qubits には勾配なし
        return grad_params, None, None, None


def quantum_expectation(
    params: torch.Tensor,
    circuit: 'QuantumCircuit',
    observable: 'PauliOperator'
) -> torch.Tensor:
    """
    量子期待値を計算する関数API
    
    Args:
        params: パラメータ [batch_size, n_params] or [n_params]
        circuit: 量子回路
        observable: 観測量
    
    Returns:
        期待値 [batch_size]
    """
    if params.dim() == 1:
        params = params.unsqueeze(0)
    
    return QuantumExpectation.apply(
        params,
        circuit.operations,
        observable.to_observable(),
        circuit.n_qubits
    )
