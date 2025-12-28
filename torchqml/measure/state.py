"""State vector access"""

import torch
import torch.nn as nn
from typing import List, Optional
import cudaq
import numpy as np

from ..core.qvector import QVector
from ..utils.kernel_builder import build_circuit_kernel


def state(qvector: QVector) -> torch.Tensor:
    """
    Get the state vector
    
    Args:
        qvector: Quantum state vector
        
    Returns:
        State vector tensor [batch_size, 2^n_qubits] (complex64/128)
    """
    operations = qvector._get_operations()
    n_qubits = qvector.n_qubits
    batch_size = qvector.batch_size
    device = qvector.device
    
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
        params_np = params.detach().cpu().numpy()
    else:
        params_np = None
        
    # Build kernel (no measurement)
    kernel, thetas = build_circuit_kernel(n_qubits, operations, measure_all=False)
    
    state_vectors = []
    
    for i in range(batch_size):
        if params_np is not None:
            p = params_np[i].tolist()
        else:
            p = []
            
        # get_state returns cudaq.State object
        s = cudaq.get_state(kernel, p)
        
        # Convert to numpy/tensor
        # State object might have .data() or be convertible to numpy
        # s is usually a list of complex numbers or numpy array
        if isinstance(s, np.ndarray):
            sv = torch.from_numpy(s)
        else:
             # Try converting list
             sv = torch.tensor(list(s), dtype=torch.complex64) # or complex128
             
        state_vectors.append(sv)
    
    return torch.stack(state_vectors).to(device)
