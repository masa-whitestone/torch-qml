
import time
import torch
import numpy as np
import pytest
from torchqml.backend import StateVector, AdjointDifferentiator, GateOp, GateType, HAS_CPP_BACKEND

def create_random_circuit(n_qubits, depth, seed=42):
    np.random.seed(seed)
    ops = []
    param_idx = 0
    
    # Simple layered ansatz: RX, RY, CNOT chain
    for _ in range(depth):
        for i in range(n_qubits):
            # RX
            op = GateOp(GateType.RX, [i], param_index=param_idx)
            ops.append(op)
            param_idx += 1
            
            # RY
            op = GateOp(GateType.RY, [i], param_index=param_idx)
            ops.append(op)
            param_idx += 1
            
        # CNOT entangling
        for i in range(0, n_qubits - 1):
            op = GateOp(GateType.CNOT, [i + 1], controls=[i])
            ops.append(op)
            
    n_params = param_idx
    params = np.random.uniform(0, 2*np.pi, n_params).astype(np.float32)
    
    # Observable Z_0
    observable = [(1.0, [("z", 0)])]
    
    return params, ops, observable

def run_benchmark(n_qubits, depth, n_runs=5):
    if not HAS_CPP_BACKEND:
        print("C++ backend not available! Skipping benchmark.")
        return

    params, ops, observable = create_random_circuit(n_qubits, depth)
    
    diff = AdjointDifferentiator(n_qubits)
    
    # Warmup
    _ = diff.forward_and_gradient(params, ops, observable)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = diff.forward_and_gradient(params, ops, observable)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
        
    avg_time = np.mean(times)
    print(f"Qubits: {n_qubits}, Depth: {depth} | Time: {avg_time*1000:.2f} ms")

if __name__ == "__main__":
    print("-" * 40)
    print("Benchmarking C++ Backend (Adjoint Differentiation)")
    print("-" * 40)
    
    configs = [
        (8, 10),
        (10, 10),
        (12, 10),
        (14, 10),
        (16, 10),
        (18, 10),
        (20, 10),
    ]
    
    for n_qubits, depth in configs:
        run_benchmark(n_qubits, depth)
