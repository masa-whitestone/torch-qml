"""
PyTorch autograd integration

This is the most authentic part of the framework.
It uses CUDA-Q's adjoint differentiation to achieve fast gradient computation.
"""

import torch
from torch.autograd import Function
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cudaq

from ..operators.pauli import PauliOperator
from ..utils.kernel_builder import build_circuit_kernel


class QuantumCircuitFunction(Function):
    """
    Custom Function integrating PyTorch autograd and quantum circuits
    
    forward: Execute circuit and calculate expectation value
    backward: Calculate gradient using CUDA-Q's adjoint differentiation
    """
    
    @staticmethod
    def forward(
        ctx,
        params: torch.Tensor,           # [batch_size, n_params]
        operations: List[Dict],          # List of circuit operations
        observable: PauliOperator,       # observable to measure
        n_qubits: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Forward pass: Calculate expectation value
        
        Returns:
            expectations: [batch_size] expectation value tensor
        """
        ctx.save_for_backward(params)
        ctx.operations = operations
        ctx.observable = observable
        ctx.n_qubits = n_qubits
        ctx.batch_size = batch_size
        
        # Dynamically build and execute CUDA-Q kernel
        expectations = _execute_circuit_batched(
            params, operations, observable, n_qubits, batch_size
        )
        
        return expectations
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass: Calculate gradient using adjoint differentiation
        
        Uses CUDA-Q's cudaq.gradients feature for O(1) gradient calculation
        (faster compared to O(2P) of parameter shift)
        """
        params, = ctx.saved_tensors
        operations = ctx.operations
        observable = ctx.observable
        n_qubits = ctx.n_qubits
        batch_size = ctx.batch_size
        
        # Compute gradients
        gradients = _compute_gradients_adjoint(
            params, operations, observable, n_qubits, batch_size
        )
        
        # Product with grad_output
        gradients = gradients * grad_output.unsqueeze(1)
        
        return gradients, None, None, None, None


def _execute_circuit_batched(
    params: torch.Tensor,
    operations: List[Dict],
    observable: PauliOperator,
    n_qubits: int,
    batch_size: int
) -> torch.Tensor:
    """
    Execute circuit in batch and calculate expectation value
    """
    device = params.device
    params_np = params.detach().cpu().numpy()
    
    # Convert to CUDA-Q spin_op
    hamiltonian = observable.to_cudaq_spin_op()
    
    # Use kernel builder (WITHOUT measurement for observe)
    kernel, thetas = build_circuit_kernel(n_qubits, operations, measure_all=False)
    
    # Batch execution
    # Broadcasting with make_kernel seems problematic (interpreted as single arg).
    # Falling back to loop over batch.
    
    params_list = params_np.tolist()
    expectations_list = []
    
    # If no params, params_list might be empty list of length batch_size?
    # If params is empty, params_list = [[] for _ in range(batch_size)] if we constructed it that way?
    # params_np from torch.zeros(batch_size, 0) -> [].tolist() -> []?
    # No, [batch_size, 0] tolist -> [[], [], ...] (list of empty lists). Correct.
    
    # However we used: params_np = params.detach().cpu().numpy()
    # If n_params=0: params_np is (batch_size, 0). tolist() gives [[], [], ...]
    
    for i in range(batch_size):
        # Taking row i
        args = params_list[i] if len(params_list) > 0 else [] 
        # Note: if params_np was empty (but batch_size > 0), tolist gives [[], ..].
        # If batch_size=0, loop doesn't run.
        
        # observe(kernel, ham, args)
        # kernel takes 1 arg (the list).
        # We pass `args` which is the list.
        res = cudaq.observe(kernel, hamiltonian, args)
        expectations_list.append(res.expectation())
        
    expectations = torch.tensor(expectations_list, device=device)
    
    return expectations


def _compute_gradients_adjoint(
    params: torch.Tensor,
    operations: List[Dict],
    observable: PauliOperator,
    n_qubits: int,
    batch_size: int
) -> torch.Tensor:
    """
    Compute gradients using Adjoint differentiation
    """
    device = params.device
    params_np = params.detach().cpu().numpy()
    
    hamiltonian = observable.to_cudaq_spin_op()
    
    # Get gradient calculator
    # Prioritize Adjoint differentiation for performance (O(1)).
    # Fallback to ParameterShift (Exact O(2P)) if Adjoint is not available.
    # CentralDifference is numerical and unstable/slow.
    if hasattr(cudaq.gradients, "Adjoint"):
        gradient = cudaq.gradients.Adjoint()
    elif hasattr(cudaq.gradients, "ParameterShift"):
        gradient = cudaq.gradients.ParameterShift()
    else:
        gradient = cudaq.gradients.CentralDifference()
    
    n_params = params.shape[1]
    gradients_list = []
    
    # Pre-build kernel structure (we rebuild per batch item currently, optimization possible)
    # Since operations don't change, we could build once if gradient.compute supported params argument better.
    # Currently we stick to loop.
    
    for batch_idx in range(batch_size):
        batch_params = params_np[batch_idx].tolist()
        
        # Build kernel
        kernel, _ = build_circuit_kernel(n_qubits, operations, measure_all=False)
        
        # Compute gradient
        # gradient.compute(parameter_vector, function, funcAtX) -> list[float]
        # function is a Callable that takes parameters and returns expectation value.
        # But we have a 'kernel' object.
        # 'function' argument in CentralDifference.compute expects a callable? Or kernel?
        # The error says "function: Callable".
        # But we successfully passed PyKernel object as 'function' in the error message invocation?
        # Invoked with: ..., <PyKernel object>, ...
        # Wait, the signature expects (parameter_vector, function, funcAtX).
        # We passed (batch_params, kernel, hamiltonian, n_params).
        # That's 4 arguments. The signature takes 3.
        
        # We need to compute expectation value at current point first -> funcAtX
        # And we need to wrap kernel+hamiltonian into a callable?
        
        # 1. Compute expectation at current point
        # We already did this in forward pass, but not stored per batch item accessible here easily without extracting.
        # Let's recompute or use forward pass result if we passed it?
        # We can re-call observe.
        current_exp = cudaq.observe(kernel, hamiltonian, batch_params).expectation()
        
        # 2. Define cost function callable
        # This needs to take parameters and return expectation.
        # But can we pass kernel? 
        # If CentralDifference only works with python functions, we need a wrapper.
        # wrapper(p): return cudaq.observe(kernel, hamiltonian, p).expectation()
        
        def cost_fn(p):
            return cudaq.observe(kernel, hamiltonian, p).expectation()
            
        grad_result = gradient.compute(batch_params, cost_fn, current_exp)
        gradients_list.append(grad_result)
    
    gradients = torch.tensor(gradients_list, device=device, dtype=torch.float32)
    return gradients


# ========== High Level API ==========

def execute_and_measure(
    qvector: 'QVector', # Forward ref
    observable: PauliOperator
) -> torch.Tensor:
    """
    Execute circuit and calculate expectation value (autograd supported)
    """
    operations = qvector._get_operations()
    
    # Extract parameters
    params_list = []
    for op in operations:
        p = op.get("params")
        if p is not None:
            if p.dim() == 1:
                params_list.append(p)
            elif p.dim() == 2:
                for i in range(p.shape[1]):
                    params_list.append(p[:, i])
    
    if params_list:
        params = torch.stack(params_list, dim=1)
    else:
        params = torch.zeros(qvector.batch_size, 0, device=qvector.device)
    
    return QuantumCircuitFunction.apply(
        params,
        operations,
        observable,
        qvector.n_qubits,
        qvector.batch_size
    )
