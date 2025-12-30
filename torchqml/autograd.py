"""
PyTorch autograd統合
"""

import torch
from torch.autograd import Function
import numpy as np

from .backend import AdjointDifferentiator, GateOp, HAS_CPP_BACKEND


class QuantumExpectation(Function):
    @staticmethod
    def forward(ctx, params, operations, observable, n_qubits):
        device = params.device
        # C++ backend expects float32 numpy on CPU (it handles transfer to GPU internally or expects pointers)
        # In this impl, we pass CPU numpy to C++.
        params_np = params.detach().cpu().numpy().astype(np.float32) 
        # If params is batch, shape (batch, n_params)
        if params_np.ndim == 1:
            params_np = params_np.reshape(1, -1)
        
        diff = AdjointDifferentiator(n_qubits)
        
        if params.requires_grad:
            exp_vals, grads = diff.forward_and_gradient(
                params_np, operations, observable
            )
            # grads returned as numpy
            ctx.save_for_backward(torch.from_numpy(grads).to(device))
        else:
            exp_vals = diff.forward_only(params_np, operations, observable)
        
        ctx.device = device
        result = torch.from_numpy(exp_vals).float().to(device)
        return result.squeeze() if result.numel() == 1 else result
    
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (batch_size)
        if ctx.needs_input_grad[0]:
            grads, = ctx.saved_tensors
            # grads: (batch_size, n_params)
            # grad_output: (batch_size) or scalar
            
            if grad_output.ndim == 0:
                grad_output = grad_output.unsqueeze(0)
                
            if grads.shape[0] != grad_output.shape[0]:
                 # Broadcasting check or error
                 pass

            # Chain rule: dL/dParam = dL/dExp * dExp/dParam
            # element-wise multiply for batch
            grad_input = grads * grad_output.unsqueeze(1)
            return grad_input, None, None, None
        return None, None, None, None

def quantum_expectation(params, operations, observable, n_qubits):
    return QuantumExpectation.apply(params, operations, observable, n_qubits)
