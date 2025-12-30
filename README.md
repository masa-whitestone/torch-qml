# torchqml

PyTorch Quantum Machine Learning with cuQuantum - Fast Adjoint Differentiation

## Features

- **Adjoint Differentiation**: O(G) gradient calculation using cuQuantum's `custatevec`.
- **PyTorch Native**: Fully integrated with `torch.autograd` and `torch.nn`.
- **Pure Python**: No C++ build required, utilizing `cupy` and `cuquantum-python`.

## Installation

```bash
pip install .
```

## Usage

```python
import torch
import torchqml as tq

# Build circuit
qc = tq.QuantumCircuit(2)
qc.h(0)
qc.ry(0, param_index=0)
qc.cx(0, 1)

# Parameters
params = torch.tensor([[0.5]], requires_grad=True)

# Expectation
exp_val = qc.expectation(params, tq.Z(0))

# Backward
exp_val.backward()
print(params.grad)
```
