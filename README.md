# torchqml

PyTorch Quantum Machine Learning with cuQuantum - Fast Adjoint Differentiation

## Features

- **Adjoint Differentiation**: O(G) gradient calculation using cuQuantum's `custatevec` and optimized C++ kernels.
- **PyTorch Native**: Fully integrated with `torch.autograd` and `torch.nn`.
- **High Performance**: Custom C++/CUDA extension reusing cuStateVec/cuBLAS handles for minimize overhead.

## Performance
TorchQML outperforms PennyLane's `lightning.gpu` backend significantly on NVIDIA GPUs (measured on T4):

| Qubits | Layers | TorchQML (ms) | PennyLane (ms) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 4 | 5 | 145.27 | 398.54 | **2.7x** |
| 8 | 10 | 251.39 | 644.85 | **2.6x** |
| 12 | 10 | 468.82 | 961.92 | **2.1x** |

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
