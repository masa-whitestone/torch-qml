"""Sampling measurement"""

import torch
from typing import Dict, List, Optional
import cudaq
import collections

from ..core.qvector import QVector
from ..utils.kernel_builder import build_circuit_kernel


def sample(
    qvector: QVector,
    shots: int = 1000
) -> List[Dict[str, int]]:
    """
    Sample from the quantum state
    
    Args:
        qvector: Quantum state vector
        shots: Number of shots
        
    Returns:
        List of count dictionaries (one per batch item)
        e.g. [{"00": 50, "11": 50}, ...]
    """
    operations = qvector._get_operations()
    n_qubits = qvector.n_qubits
    batch_size = qvector.batch_size
    
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
        params = torch.stack(params_list, dim=1) # [batch_size, n_params]
        params_args = params.detach().cpu().numpy().tolist()
    else:
        # For batching with no parameters, we might need to pass list of empty lists?
        # Or just empty list? cudaq.sample(kernel, [[]] * batch_size)?
        params_args = [[] for _ in range(batch_size)]
    
    # Build kernel WITH measurement
    kernel, thetas = build_circuit_kernel(n_qubits, operations, measure_all=True)
    
    results = []
    
    try:
        # Try batch execution
        # If params_args is [[], []], does it imply batch size 2?
        sample_results = cudaq.sample(kernel, params_args, shots_count=shots)
        
        if batch_size == 1:
             # If single item batch, sample returns SampleResult directly?
             # Or if we passed [[...]], it returns [SampleResult]?
             # If params_args is [], it treats as single?
             # Let's handle return type safely
             if hasattr(sample_results, 'items'):
                 results.append(dict(sample_results.items()))
             else:
                 # Assume iterable
                 for res in sample_results:
                     results.append(dict(res.items()))
        else:
            for res in sample_results:
                results.append(dict(res.items()))
                
    except Exception as e:
        # Fallback to loop
        for i in range(batch_size):
            p = params_args[i]
            res = cudaq.sample(kernel, p, shots_count=shots)
            results.append(dict(res.items()))
            
    return results
