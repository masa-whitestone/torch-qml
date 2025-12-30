import time
import torch
import numpy as np
import torchqml as tq

# PennyLane (比較用)
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def benchmark_torchqml(n_qubits, n_layers, n_iterations=10):
    """torchqmlベンチマーク"""
    n_params = n_qubits * n_layers
    
    # 回路構築
    qc = tq.QuantumCircuit(n_qubits)
    param_idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(q, param_index=param_idx)
            param_idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    
    params = torch.randn(1, n_params, requires_grad=True)
    
    # Warmup
    exp_val = qc.expectation(params, tq.Z(0))
    exp_val.backward()
    
    # Benchmark
    times = []
    for _ in range(n_iterations):
        params.grad = None
        t0 = time.time()
        exp_val = qc.expectation(params, tq.Z(0))
        exp_val.backward()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    
    return np.mean(times) * 1000  # ms


def benchmark_pennylane(n_qubits, n_layers, n_iterations=10):
    """PennyLaneベンチマーク"""
    if not HAS_PENNYLANE:
        return None
    
    n_params = n_qubits * n_layers
    dev = qml.device("lightning.gpu", wires=n_qubits)
    
    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        params = params.reshape(n_layers, n_qubits)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(params[layer, q], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0))
    
    # PennyLaneのnumpyを使用
    import pennylane.numpy as pnp
    params = pnp.random.randn(n_params, requires_grad=True)
    
    # Warmup
    _ = circuit(params)
    _ = qml.grad(circuit)(params)
    
    # Benchmark
    times = []
    for _ in range(n_iterations):
        t0 = time.time()
        _ = circuit(params)
        res = qml.grad(circuit)(params) # resは勾配
        # PennyLaneの非同期実行を考慮して待機（lightning.gpuは通常同期だが念のため）
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
        times.append(time.time() - t0)
    
    return np.mean(times) * 1000  # ms


if __name__ == "__main__":
    configs = [
        (4, 5),
        (8, 10),
        (12, 10),
        # (16, 10), # Too large for quick test
    ]
    
    print("=" * 60)
    print(f"{'n_qubits':>10} {'n_layers':>10} {'torchqml':>15} {'PennyLane':>15}")
    print("=" * 60)
    
    for n_qubits, n_layers in configs:
        try:
            t_tq = benchmark_torchqml(n_qubits, n_layers)
            t_tq_str = f"{t_tq:.2f} ms"
        except Exception as e:
            t_tq_str = f"Error: {e}"

        try:
            t_pl = benchmark_pennylane(n_qubits, n_layers)
            pl_str = f"{t_pl:.2f} ms" if t_pl else "N/A"
        except Exception as e:
             pl_str = f"Error: {e}"
        
        print(f"{n_qubits:>10} {n_layers:>10} {t_tq_str:>15} {pl_str:>15}")
    
    print("=" * 60)
