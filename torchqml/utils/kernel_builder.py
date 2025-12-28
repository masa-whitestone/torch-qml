"""Kernel building utilities"""

import cudaq
from typing import List, Dict, Tuple, Any
import cudaq
from typing import List, Dict, Tuple, Any
import hashlib
import json

# Gate imports
from ..gates.single import h, x, y, z, s, t
from ..gates.parametric import rx, ry, rz, r1



# Global cache
_KERNEL_CACHE = {}

def _hash_operations(n_qubits: int, operations: List[Dict], measure_all: bool) -> str:
    """Create a hash for the circuit structure"""
    # Create a string representation of the circuit structure
    # We only care about gate types, targets, controls, adjoint.
    # Parameters are excluded.
    structure = [str(n_qubits), str(measure_all)]
    for op in operations:
        if isinstance(op, tuple):
             gate, targets, controls, adjoint, p_idx = op
             op_str = f"{gate}:{targets}:{controls}:{adjoint}"
        else:
            op_str = f"{op['gate']}:{op['targets']}:{op.get('controls')}:{op.get('adjoint')}"
        structure.append(op_str)
    
    struct_str = "|".join(structure)
    return hashlib.md5(struct_str.encode()).hexdigest()

def build_circuit_kernel(
    n_qubits: int, 
    operations: List[Dict],
    measure_all: bool = False,
    use_cache: bool = True
) -> Tuple[Any, Any]:
    """
    Build a CUDA-Q kernel from a list of operations.
    
    Args:
        n_qubits: Number of qubits
        operations: List of operation dictionaries
        measure_all: If True, add measurement to all qubits at the end
        use_cache: If True, try to use cached kernel
        
    Returns:
        (kernel, thetas) tuple where kernel is the cudaq kernel object 
        and thetas is the parameter list argument
    """
    if use_cache:
        cache_key = _hash_operations(n_qubits, operations, measure_all)
        if cache_key in _KERNEL_CACHE:
            return _KERNEL_CACHE[cache_key]

    kernel, thetas = cudaq.make_kernel(list)
    qubits = kernel.qalloc(n_qubits)
    
    param_idx = 0

    
    # instructions: List[Tuple(name, targets, controls, adjoint, param_idx)]
    for inst in operations:
        # Check if it's new tuple format or old dict
        if isinstance(inst, tuple):
            gate, targets, controls, adjoint, p_idx = inst
        else:
            # Backward compatibility (though we refactored everything)
            gate = inst.get("gate")
            targets = inst.get("targets")
            controls = inst.get("controls")
            adjoint = inst.get("adjoint", False)
        
        # ... logic ...
        
        # Handle SWAP special case
        if gate == "swap":
            if len(targets) == 2:
                # SWAP doesn't usually support controls in basic mapping unless decomposed
                # Ignoring controls for SWAP for now as per previous implementation
                kernel.swap(qubits[targets[0]], qubits[targets[1]])
            continue

        for target in targets:
            if gate == "h":
                if controls:
                    kernel.ch(qubits[controls[0]], qubits[target])
                else:
                    kernel.h(qubits[target])
            elif gate == "x":
                if controls:
                    # Support multiple controls?
                    if len(controls) == 1:
                        kernel.cx(qubits[controls[0]], qubits[target])
                    elif len(controls) == 2:
                        kernel.ccx(qubits[controls[0]], qubits[controls[1]], qubits[target])
                    else:
                        # Fallback or error?
                        # For now, assume simple controls or support limited
                         kernel.cx(qubits[controls[0]], qubits[target])
                else:
                    kernel.x(qubits[target])
            elif gate == "y":
                if controls:
                    kernel.cy(qubits[controls[0]], qubits[target])
                else:
                    kernel.y(qubits[target])
            elif gate == "z":
                if controls:
                    kernel.cz(qubits[controls[0]], qubits[target])
                else:
                    kernel.z(qubits[target])
            elif gate == "s":
                if controls:
                    # CS not always standard in basic API, might need decomposition
                    # But if cudaq supports it:
                    # kernel.cs(c, t)? 
                    # Let's stick to what we had or simple application
                    pass 
                else:
                    kernel.s(qubits[target])
            elif gate == "t":
                if controls:
                    pass
                else:
                    kernel.t(qubits[target])
            elif gate == "rx":
                sign = -1.0 if adjoint else 1.0
                kernel.rx(sign * thetas[param_idx], qubits[target])
                param_idx += 1
            elif gate == "ry":
                sign = -1.0 if adjoint else 1.0
                kernel.ry(sign * thetas[param_idx], qubits[target])
                param_idx += 1
            elif gate == "rz":
                sign = -1.0 if adjoint else 1.0
                kernel.rz(sign * thetas[param_idx], qubits[target])
                param_idx += 1
            elif gate == "r1":
                sign = -1.0 if adjoint else 1.0
                kernel.r1(sign * thetas[param_idx], qubits[target])
                param_idx += 1
            elif gate == "u3":
                # U3(theta, phi, lambda)
                # thetas list is flattened. 
                # We need to take 3 params.
                t_val = thetas[param_idx]
                p_val = thetas[param_idx+1]
                l_val = thetas[param_idx+2]
                # cudaq.u3?
                # If not available, decompose: Rz(phi) Ry(theta) Rz(lambda) ?
                # Or Kernel.u3 if exposed.
                # Checking docs or assuming decomposition.
                # U3(theta, phi, lambda) = Rz(phi + 3pi) Rx(pi/2) Rz(theta + pi) Rx(pi/2) Rz(lambda) ... no
                # Standard decomposition: Rz(phi) Ry(theta) Rz(lambda) (up to phase?)
                # Actually U3(theta, phi, lam) ~ Rz(phi) * Ry(theta) * Rz(lam)
                # Let's try kernel.u3 if it exists, otherwise leave simple implementation TODO or approximate
                try:
                    kernel.u3(t_val, p_val, l_val, qubits[target])
                except AttributeError:
                    # Fallback decomposition if kernel.u3 missing
                    kernel.rz(l_val, qubits[target])
                    kernel.ry(t_val, qubits[target])
                    kernel.rz(p_val, qubits[target])
                
                param_idx += 3

    if measure_all:
        kernel.mz(qubits)
        
    if use_cache:
        _KERNEL_CACHE[cache_key] = (kernel, thetas)
        
    return kernel, thetas
