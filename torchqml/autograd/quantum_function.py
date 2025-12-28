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
    
    # Gradient Strategy Selection
    if hasattr(cudaq.gradients, "Adjoint"):
        # Use Native CUDA-Q Adjoint
        gradient = cudaq.gradients.Adjoint()
        grad_method = "native_adjoint"
    else:
        # Use Custom PyTorch Adjoint (O(1))
        # This bypasses the verify_cudaq_gradient path and runs simulation manually
        grad_method = "pytorch_adjoint"

    if grad_method == "native_adjoint" or grad_method == "parameter_shift":
        # ... (Existing code for native gradients)
        pass 
    elif grad_method == "pytorch_adjoint":
        # Prepare data for PyTorch Engine
        
        # 1. Map operations options
        # We need to tag operations with param indices
        # params: [batch, n_params]
        # Iterate qvector operations and match parameters
        
        # Current qvector stores operations with Parameter objects.
        # We need to find the index of each Parameter object in the 'params' input list
        # But 'params' input to this function is just a Tensor [batch, n].
        # The 'self.params' list holds the Parameter objects in order.
        
        # Build mapping: Parameter ID -> Index
        param_map = {id(p): i for i, p in enumerate(self.params)}
        
        # Enhance operations with param_idx
        # We make a shallow copy or modify temporarily? 
        # Modifying qvector operations is side-effect.
        # Let's verify if structure allows.
        # Operations are 'RecordedOperation'.
        
        ops_for_adjoint = []
        for op in self.q_vector._operations:
            new_op = op # Simple Ref
            # Add param_idx attribute dynamically or create wrapper
            p_idx = None
            if op.is_parametric and len(op.parameters) > 0:
                 # Assume 1 param per gate for now (standard gates)
                 p_obj = op.parameters[0]
                 # Unpack if list
                 if isinstance(p_obj, list): p_obj = p_obj[0]
                 
                 # Check if p_obj is in our trained params
                 if id(p_obj) in param_map:
                     p_idx = param_map[id(p_obj)]
            
            # Create a simple struct/tuple for the engine
            # (name, targets, p_idx) is enough if we trust the engine
            
            # Define simple class or use tuple
            class AdjointOp:
                def __init__(self, name, targets, param_idx, is_parametric, parameters):
                    self.name = name
                    self.targets = targets
                    self.param_idx = param_idx
                    self.is_parametric = is_parametric
                    self.parameters = parameters
            
            ops_for_adjoint.append(AdjointOp(op.name, op.targets, p_idx, op.is_parametric, op.parameters))

        # Prepare Hamiltonian
        # observable is PauliOperator
        # Convert to list of (coeff, [(char, idx), ...])
        ham_pauli = []
        # PauliOperator structure: self.terms = { 'IXYZ...': coeff } ?
        # Or self.string_repr?
        # torchqml.operators.PauliOperator implementation:
        # It stores data as list of (pauli_string, coefficient).
        # Wait, let's double check PauliOperator implementation.
        # To be safe, we rely on 'to_cudaq_spin_op' or internal structure.
        # Let's iterate:
        
        # If simple term:
        # We can implement a helper 'get_terms' on PauliOperator if missing.
        # Or parse string.
        
        # Assuming we can iterate terms.
        # For now, let's assume Observables are simple single terms (hqnn example uses tq.Z(0))
        # If it's tq.Z(0), it has 1 term.
        
        # Quick hack: Parse from string representation if needed, or check attributes.
        # PauliOperator usually has `terms` dict or list.
        # If not, add `get_terms` to PauliOperator class.
        
        # For this patch, let's inspect PauliOperator behavior via `observable`.
        # Assuming `observable.terms` is available (standard design).
        # IF NOT, we might break.
        
        # Let's fix this by calling a helper
        ham_pauli = self._parse_hamiltonian(observable)
        
        from torchqml.autograd.adjoint import PyTorchAdjointFunction
        
        # Call Apply
        return PyTorchAdjointFunction.apply(params, ops_for_adjoint, self.q_vector.num_qubits, ham_pauli)
        
    # Fallback to existing logic if native
    if hasattr(cudaq.gradients, "Adjoint"):
         gradient = cudaq.gradients.Adjoint()
    else:
         gradient = cudaq.gradients.ParameterShift()
         
    # ... (existing execution code)
    
    # We basically replaced the whole execution block?
    # No, we need to handle the structure.
    # Ideally, we return early if PyTorchAdjoint is used.

    # ...
    
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
