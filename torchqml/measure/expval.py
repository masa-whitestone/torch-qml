"""Expectation value calculation"""

import torch
from typing import Union, List

from ..core.qvector import QVector
from ..operators.pauli import PauliOperator
from ..autograd.quantum_function import execute_and_measure


def expval(
    qvector: QVector,
    observable: Union[PauliOperator, List[PauliOperator]]
) -> torch.Tensor:
    """
    Calculate expectation value
    
    Args:
        qvector: Quantum state vector (with operations recorded)
        observable: observable to measure, or list of observables
    
    Returns:
        Single observable: [batch_size]
        Multiple observables: [batch_size, n_observables]
    """
    if isinstance(observable, list):
        # Multiple observables
        results = []
        for obs in observable:
            result = execute_and_measure(qvector, obs)
            results.append(result)
        return torch.stack(results, dim=1)
    else:
        # Single observable
        return execute_and_measure(qvector, observable)
