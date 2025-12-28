import torch
import torchqml as tq
import numpy as np

def apply_gate(state, gate_matrix, targets, n_qubits):
    """
    Applies a gate matrix to the state vector.
    state: [batch, dim]
    gate_matrix: [batch, 2^k, 2^k] or [2^k, 2^k]
    targets: list of target qubit indices
    """
    batch_size = state.shape[0]
    # Check if gate_matrix has batch dim
    if gate_matrix.ndim == 2:
        gate_matrix = gate_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
    k = len(targets)
    
    # Reshape state to access target qubits
    # shape: [batch, 2, 2, ..., 2] (n_qubits + 1 dims)
    # We want to bring target axes to the end for matmul
    
    # Example: 1 qubit gate on qubit i
    # reshape [batch, 2^(n-1-i), 2, 2^i] -> permute -> matmul
    # This tensor manipulation is standard in state vector simulators.
    
    # Native PyTorch implementation:
    # 1. View as [batch, 2, 2, ..., 2]
    # 2. Permute target dimensions to the end
    # 3. Flatten non-target dimensions
    # 4. Matmul
    # 5. Reshape back and permute back
    
    state_view = state.view([batch_size] + [2]*n_qubits)
    
    # Compute permutation
    # We want [batch, others, targets]
    # Indices in view correspond to [q_{n-1}, ..., q_0] or [q_0, ..., q_{n-1}]?
    # Usually q0 is LSB or MSB. Let's assume q0 is LSB (last dim).
    # Then target i is dimension -(i+1).
    # Let's verify qvector convention.
    # If q0 is index 0 in list, and we use Kronecker product convention:
    # |q0 q1 ...> ?
    # Standard is usually |q_{n-1} ... q0>.
    # Let's assume indices match the view dimensions [q0, q1, ... q_{n-1}] for simplicity, 
    # and map gates accordingly.
    # view dims: 1 (batch), 2 (q0), 3 (q1) ...
    
    # Dimensions: 0=batch, 1=q0, 2=q1, ... n=q{n-1}
    all_dims = list(range(1, n_qubits + 1))
    target_dims = [t + 1 for t in targets]
    other_dims = [d for d in all_dims if d not in target_dims]
    
    perm = [0] + other_dims + target_dims
    
    # Permute
    state_perm = state_view.permute(perm)
    
    # Flatten for matmul
    # [batch, 2^(n-k), 2^k]
    others_size = 2**(n_qubits - k)
    targets_size = 2**k
    
    state_flat = state_perm.reshape(batch_size, others_size, targets_size)
    
    # Matmul: gate [batch, 2^k, 2^k]
    # We need: state * gate^T  (since state is row vectors? No, usually column vectors)
    # If state view is [batch, ..., targets], this effectively treats last dims as vector components.
    # To apply G |psi>:
    # We want vector v' = G v.
    # In tensor form: v'_{...i...} = Sum_j G_{ij} v_{...j...}
    # Here, we have state_flat as [batch, others, targets].
    # This looks like a batch of row vectors if we view last dim as 'targets'.
    # Matmul: state_flat @ gate_matrix.mT (transpose)
    
    # Gate matrix construction:
    # Single qubit gate on q0: I x ... x U
    # This implementation handles 'targets' as the specific subspace.
    
    new_state_flat = torch.matmul(state_flat, gate_matrix.transpose(1, 2))
    
    # Reshape back
    new_state_perm = new_state_flat.reshape([batch_size] + [2]*(n_qubits-k) + [2]*k)
    
    # Inverse Permute
    # We need to compute inv_perm to restore order
    inv_perm = [0] * (n_qubits + 1)
    for i, p in enumerate(perm):
        inv_perm[p] = i
        
    new_state = new_state_perm.permute(inv_perm)
    return new_state.reshape(batch_size, -1)

def get_gate_matrix_tensor(name, param, device):
    """Factory for gate matrices (differentiable)."""
    if name == 'rx':
        # RX(theta) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]
        theta = param
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        zero = torch.zeros_like(c)
        # [[c, -is], [-is, c]]
        # Stack: [ [c, -is], [-is, c] ]
        # Shape: [batch, 2, 2]
        row1 = torch.stack([c + 0j, -1j * s], dim=1)
        row2 = torch.stack([-1j * s, c + 0j], dim=1)
        return torch.stack([row1, row2], dim=1)
    
    elif name == 'ry':
        # RY(theta) = [[cos, -sin], [sin, cos]]
        theta = param
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        row1 = torch.stack([c + 0j, -s + 0j], dim=1)
        row2 = torch.stack([s + 0j, c + 0j], dim=1)
        return torch.stack([row1, row2], dim=1)
        
    elif name == 'rz':
        # RZ(theta) = [[e^{-it/2}, 0], [0, e^{it/2}]]
        theta = param
        e_neg = torch.exp(-0.5j * theta)
        e_pos = torch.exp(0.5j * theta)
        zero = torch.zeros_like(theta, dtype=torch.complex128)
        row1 = torch.stack([e_neg, zero], dim=1)
        row2 = torch.stack([zero, e_pos], dim=1)
        return torch.stack([row1, row2], dim=1)
    
    elif name == 'h':
        val = 1 / np.sqrt(2)
        mat = torch.tensor([[val, val], [val, -val]], dtype=torch.complex128, device=device)
        return mat
        
    elif name == 'x':
        mat = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
        return mat
        
    elif name == 'y':
        mat = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
        return mat

    elif name == 'z':
        mat = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
        return mat
        
    elif name == 'cnot' or name == 'cx':
        # 4x4 matrix
        mat = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex128, device=device)
        return mat
    
    # Fallback/TODO: Add more gates
    return torch.eye(2, dtype=torch.complex128, device=device)

class PyTorchAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, operations, n_qubits, hamiltonian_pauli):
        """
        params: [batch, n_params]
        operations: list of (gate_name, targets, param_index/None) tuples
        hamiltonian_pauli: list of (pauli_string, coeff) - Simplified for now
                           Actually we need to support generic Observable. 
                           Expect normalized PauliOperator for now.
        """
        device = params.device
        batch_size = params.shape[0]
        dim = 2**n_qubits
        
        # 1. Init state
        state = torch.zeros((batch_size, dim), dtype=torch.complex128, device=device)
        state[:, 0] = 1.0 + 0j
        
        intermediate_states = [state] # state before op 0
        
        # 2. Forward Simulation
        # We need to map operations to matrices
        # And track which param index is used
        
        saved_ops = [] # (name, targets, matrix, param_idx)
        
        current_state = state
        
        for op in operations:
            name = op.name
            targets = [t.index if isinstance(t, tq.Qubit) else t for t in op.targets]
            
            # Determine Matrix
            mat = None
            p_idx = None
            
            # Check if parameterized
            if op.is_parametric and len(op.parameters) > 0:
                # Find which param from 'params' tensor corresponds to this op
                # In torchqml, parameters are passed as list of Parameter objects.
                # 'params' arg here is the Tensor passed to apply().
                # We assume 'operations' struct allows identifying the param.
                # IMPORTANT: The 'params' tensor passed to forward is just values.
                # We need mapping.
                
                # For this implementation, we assume 'operations' contains a 'param_index' 
                # that points to the column in 'params'.
                # We need to preprocess operations to add this index before calling apply.
                
                if hasattr(op, 'param_idx'):
                    p_idx = op.param_idx
                    # Get param for batch
                    p_val = params[:, p_idx]
                    mat = get_gate_matrix_tensor(name, p_val, device)
                else:
                    # Non-parametric usage of parametric gate? Or constant param.
                    # Handle constant params later. For now assume all are trained.
                    mat = get_gate_matrix_tensor(name, torch.zeros(batch_size, device=device), device)
            else:
                mat = get_gate_matrix_tensor(name, None, device)
            
            # Apply
            current_state = apply_gate(current_state, mat, targets, n_qubits)
            intermediate_states.append(current_state)
            saved_ops.append((name, targets, mat, p_idx))
            
        ctx.saved_ops = saved_ops
        ctx.intermediate_states = intermediate_states
        ctx.hamiltonian_pauli = hamiltonian_pauli
        ctx.n_qubits = n_qubits
        
        # 3. Measurement <psi|H|psi>
        # Assume H = Z0 for simplicity test, or parse Hamiltonian
        # H should be a Sparse Matrix or sum of Paulis
        
        # Let's perform measurement using matrix
        # Construct H matrix
        # For simplicity, let's assume H is a single Pauli term or sum.
        # We construct H_mat on fly for now (inefficient for large N, but ok for test)
        
        # Construct H matrix from Pauli string
        # ... (Implementation detail: Kronecker product)
        
        # For Optimization: Only apply H to kets. H|psi>.
        # Then overlap.
        
        # Placeholder H: Z on qubit 0
        # If user passed tq.Z(0), we apply Z matrix to q0.
        
        # Parse hamiltonian string (e.g. "1.0 Z0")
        # Assume hamiltonian_pauli is a list of terms [(coeff, [(pauli_char, qubit_idx), ...])]
        
        total_expval = torch.zeros(batch_size, dtype=torch.float64, device=device)
        
        # We need H|psi>
        # Let's compute phi = H|psi>
        # Since H is hermitian, expval = <psi|phi>
        
        # We can implement 'apply_hamiltonian' similar to apply_gate
        psi = current_state
        phi = torch.zeros_like(psi)
        
        for term in hamiltonian_pauli:
            coeff = term[0]
            ops = term[1] # list of (char, idx)
            
            term_psi = psi.clone()
            for char, idx in ops:
                 mat = get_gate_matrix_tensor(char.lower(), None, device)
                 term_psi = apply_gate(term_psi, mat, [idx], n_qubits)
            
            phi += coeff * term_psi
            
        # Expval = real(<psi|phi>)
        # psi: [batch, dim], phi: [batch, dim]
        # batch dot product
        expval = torch.sum(torch.conj(psi) * phi, dim=1).real
        
        ctx.final_phi = phi # Store H|psi> for backward
        
        return expval

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [batch]
        Returns gradients w.r.t params: [batch, n_params]
        """
        saved_ops = ctx.saved_ops
        intermediate_states = ctx.intermediate_states
        phi = ctx.final_phi # lambda state initially H|psi>
        
        # Adjoint Loop
        # We have states |psi_0>, |psi_1>, ..., |psi_N>
        # We have |lambda_N> = H |psi_N> (which is phi)
        # We iterate k from N-1 down to 0
        
        # grad_params = zeros
        grad_params = None # Will init on first parametric op
        
        # We need parameter count to init grads
        # Iterate to find max index?
        # Or Just accumulate in a dict/list and stack?
        
        # Let's assume we know shape from ctx? Or grad_params shape
        # We didn't save n_params directly but can infer.
        
        grads = {} # idx -> tensor [batch]
        
        # Current lambda is phi = H|psi_N> (unnormalized gradient signal)
        lam = phi 
        
        # Backprop
        for k in range(len(saved_ops) - 1, -1, -1):
            name, targets, mat, p_idx = saved_ops[k]
            psi_prev = intermediate_states[k] # |psi_{k}> input to this gate
            
            # 1. Update lambda: |lambda_{k}> = U_k^dagger |lambda_{k+1}>
            # Apply adjoint gate to lambda
            mat_adj = mat.conj().transpose(1, 2)
            lam = apply_gate(lam, mat_adj, targets, ctx.n_qubits)
            
            # 2. If parametric, compute deriv
            if p_idx is not None:
                # dE = 2 * Real( <lambda_{k+1} | dU/dt | psi_k> )
                # But notice: lam here is ALREADY |lambda_k> = U^dag |lambda_{k+1}>
                # So we can effectively use <lambda_k | U^dag dU | psi_k> ?
                # Or standard form: <lambda_{k+1} | dU | psi_k>
                
                # We need U dU/dt?
                # For rotation gates R(t) = exp(-i t/2 G), dR/dt = -i/2 G R(t)
                # So dU | psi_k > = -i/2 G | psi_{k+1} >
                # Then <lambda_{k+1} | dU | psi_k> = <lambda_{k+1} | -i/2 G U | psi_k >
                # = <lambda_{k+1} | -i/2 G | psi_{k+1} >
                # = -i/2 <lambda_{k+1} | G | psi_{k+1} >
                
                # Alternatively: <lambda_{k} | U^dag (-i/2 G U) | psi_k >
                # = <lambda_k | -i/2 G | psi_k>  (since vectors rotate together? No)
                # U commutes with G? Yes for Pauli rotations. U = exp(..G), so they commute.
                # So U^dag G U = G.
                # Thus gradient = 2 * Real( -i/2 <lambda_k | G | psi_k> )
                # = Im( <lambda_k | G | psi_k> )
                
                # G is Pauli X, Y, Z for RX, RY, RZ
                # Apply G to psi_k
                
                # Generator name
                gen_name = {'rx':'x', 'ry':'y', 'rz':'z', 'r1':'z'}.get(name)
                
                if gen_name:
                    gen_mat = get_gate_matrix_tensor(gen_name, None, lam.device)
                    G_psi = apply_gate(psi_prev, gen_mat, targets, ctx.n_qubits)
                    
                    # Overlap
                    overlap = torch.sum(torch.conj(lam) * G_psi, dim=1)
                    
                    # Grad = - Im(overlap) ?
                    # dU = -i/2 G U. 
                    # Term = <lambda_{k+1} | -i/2 G | psi_{k+1}> = 0.5 * (-i) * <l|G|p>
                    # 2 * Real(Term) = Real( -i <l|G|p> ) = Im( <l|G|p> )
                    
                    # Wait, rotation def: Rz(t) = exp(-i t/2 Z)
                    # d/dt = -i/2 Z Rz
                    # contribution = <L | dU | psi_in> = <L | -i/2 Z | psi_out>
                    # = -i/2 <L | Z | psi_out>
                    # Add c.c. term -> 2 Re(...)
                    # 2 * Re( -i/2 <L|Z|psi_out> ) = Re( -i <L|Z|psi_out> ) = Im( <L|Z|psi_out> )
                    
                    # Here 'lam' is updated to |lambda_k> (before gate).
                    # 'psi_prev' is |psi_k> (before gate).
                    # Commutation holds: <L_{k+1} | Z | psi_{k+1}> = <L_k | Z | psi_k>
                    # So we can use current 'lam' and 'psi_prev'.
                    
                    g_val = overlap.imag
                    
                    if p_idx not in grads:
                         grads[p_idx] = g_val
                    else:
                         grads[p_idx] += g_val

        # Assemble gradients tensor
        # grad_output chain rule: grad_params * grad_output
        
        # We need to know total n_params. We can find max p_idx.
        if len(grads) > 0:
            max_idx = max(grads.keys())
            g_tensor = torch.zeros((grad_output.shape[0], max_idx + 1), dtype=torch.float64, device=lam.device)
            for idx, val in grads.items():
                g_tensor[:, idx] = val * grad_output # Chain rule
            return g_tensor, None, None, None
        
        return None, None, None, None

