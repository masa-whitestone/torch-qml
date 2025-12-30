"""
cuQuantum custatevec バックエンド

Adjoint Differentiationによる高速勾配計算を実装
"""

import numpy as np
import cupy as cp
import cuquantum
from cuquantum.bindings import custatevec as cusv
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class GateType(Enum):
    """ゲートタイプ"""
    # 非パラメトリック
    H = "h"
    X = "x"
    Y = "y"
    Z = "z"
    S = "s"
    T = "t"
    # パラメトリック
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    # 2qubit
    CNOT = "cnot"
    CZ = "cz"
    SWAP = "swap"


@dataclass
class GateOp:
    """ゲート操作を表すデータクラス"""
    gate_type: GateType
    targets: List[int]
    controls: List[int] = None
    param_index: Optional[int] = None  # パラメータインデックス（-1 if non-parametric）
    
    def __post_init__(self):
        if self.controls is None:
            self.controls = []


class StateVector:
    """
    cuQuantum custatevec を使った状態ベクトルシミュレータ
    
    GPU上で状態ベクトルを管理し、ゲート適用・期待値計算を行う
    """
    
    def __init__(self, n_qubits: int, dtype=cp.complex64, handle=None):
        """
        Args:
            n_qubits: qubit数
            dtype: データ型 (complex64 or complex128)
            handle: custatevecハンドル (Noneの場合は新規作成)
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.dtype = dtype
        
        # custatevec ハンドル
        if handle is None:
            self.handle = cusv.create()
            self._own_handle = True
        else:
            self.handle = handle
            self._own_handle = False
        
        # 状態ベクトル (GPU)
        
        # 状態ベクトル (GPU)
        self.state = cp.zeros(self.dim, dtype=dtype)
        self.state[0] = 1.0  # |0...0⟩
        
        # ゲート行列キャッシュ
        self._gate_cache: Dict[str, cp.ndarray] = {}
        self._init_gate_cache()
    
    def _init_gate_cache(self):
        """非パラメトリックゲート行列をキャッシュ"""
        dtype = self.dtype
        
        self._gate_cache['h'] = cp.array([
            [1, 1], [1, -1]
        ], dtype=dtype) / np.sqrt(2)
        
        self._gate_cache['x'] = cp.array([
            [0, 1], [1, 0]
        ], dtype=dtype)
        
        self._gate_cache['y'] = cp.array([
            [0, -1j], [1j, 0]
        ], dtype=dtype)
        
        self._gate_cache['z'] = cp.array([
            [1, 0], [0, -1]
        ], dtype=dtype)
        
        self._gate_cache['s'] = cp.array([
            [1, 0], [0, 1j]
        ], dtype=dtype)
        
        self._gate_cache['t'] = cp.array([
            [1, 0], [0, np.exp(1j * np.pi / 4)]
        ], dtype=dtype)
    
    def reset(self):
        """状態を|0...0⟩にリセット"""
        self.state.fill(0)
        self.state[0] = 1.0
    
    def copy(self) -> 'StateVector':
        """状態のコピーを作成"""
        new_sv = StateVector(self.n_qubits, self.dtype, handle=self.handle)
        new_sv.state = self.state.copy()
        # ハンドルは共有するのでnew_svでは解放しない（_own_handle=Falseになる）
        return new_sv
    
    def get_state(self) -> cp.ndarray:
        """状態ベクトルを取得"""
        return self.state.copy()
    
    def set_state(self, state: cp.ndarray):
        """状態ベクトルを設定"""
        self.state = state.copy()
    
    # ==================== ゲート行列生成 ====================
    
    def rx_matrix(self, theta: float) -> cp.ndarray:
        """RX(theta) 行列"""
        c = float(np.cos(theta / 2))
        s = float(np.sin(theta / 2))
        return cp.array([
            [c, -1j * s],
            [-1j * s, c]
        ], dtype=self.dtype)
    
    def ry_matrix(self, theta: float) -> cp.ndarray:
        """RY(theta) 行列"""
        c = float(np.cos(theta / 2))
        s = float(np.sin(theta / 2))
        return cp.array([
            [c, -s],
            [s, c]
        ], dtype=self.dtype)
    
    def rz_matrix(self, theta: float) -> cp.ndarray:
        """RZ(theta) 行列"""
        return cp.array([
            [cp.exp(-1j * theta / 2), 0],
            [0, cp.exp(1j * theta / 2)]
        ], dtype=self.dtype)
    
    def rx_derivative(self, theta: float) -> cp.ndarray:
        """d/dtheta RX(theta) 行列"""
        c = float(np.cos(theta / 2))
        s = float(np.sin(theta / 2))
        return cp.array([
            [-s / 2, -1j * c / 2],
            [-1j * c / 2, -s / 2]
        ], dtype=self.dtype)
    
    def ry_derivative(self, theta: float) -> cp.ndarray:
        """d/dtheta RY(theta) 行列"""
        c = float(np.cos(theta / 2))
        s = float(np.sin(theta / 2))
        return cp.array([
            [-s / 2, -c / 2],
            [c / 2, -s / 2]
        ], dtype=self.dtype)
    
    def rz_derivative(self, theta: float) -> cp.ndarray:
        """d/dtheta RZ(theta) 行列"""
        return cp.array([
            [-1j / 2 * cp.exp(-1j * theta / 2), 0],
            [0, 1j / 2 * cp.exp(1j * theta / 2)]
        ], dtype=self.dtype)
    
    def get_gate_matrix(self, gate_type: GateType, param: Optional[float] = None) -> cp.ndarray:
        """ゲート行列を取得"""
        name = gate_type.value
        
        if name in self._gate_cache:
            return self._gate_cache[name]
        
        if name == 'rx':
            return self.rx_matrix(param)
        elif name == 'ry':
            return self.ry_matrix(param)
        elif name == 'rz':
            return self.rz_matrix(param)
        else:
            raise ValueError(f"Unknown gate: {name}")
    
    def get_gate_derivative(self, gate_type: GateType, param: float) -> cp.ndarray:
        """ゲート微分行列を取得"""
        name = gate_type.value
        
        if name == 'rx':
            return self.rx_derivative(param)
        elif name == 'ry':
            return self.ry_derivative(param)
        elif name == 'rz':
            return self.rz_derivative(param)
        else:
            raise ValueError(f"Gate {name} has no derivative (non-parametric)")
    
    # ==================== ゲート適用 ====================
    
    def apply_gate(
        self,
        gate_matrix: cp.ndarray,
        targets: List[int],
        controls: List[int] = None,
        adjoint: bool = False
    ):
        """
        ゲートを状態ベクトルに適用
        
        Args:
            gate_matrix: ゲート行列 [2^k, 2^k]
            targets: ターゲットqubitインデックス
            controls: 制御qubitインデックス（オプション）
            adjoint: Trueならゲートのadjoint（随伴）を適用
        """
        if controls is None:
            controls = []
        
        if adjoint:
            matrix = cp.ascontiguousarray(gate_matrix.conj().T)
        else:
            matrix = cp.ascontiguousarray(gate_matrix)
        
        n_targets = len(targets)
        n_controls = len(controls)
        
        # custatevec データ型
        if self.dtype == cp.complex64:
            cuda_dtype = cuquantum.cudaDataType.CUDA_C_32F
            compute_type = cuquantum.ComputeType.COMPUTE_32F
        else:
            cuda_dtype = cuquantum.cudaDataType.CUDA_C_64F
            compute_type = cuquantum.ComputeType.COMPUTE_64F
        
        # ワークスペースサイズ計算
        workspace_size = cusv.apply_matrix_get_workspace_size(
            self.handle,
            cuda_dtype,
            self.n_qubits,
            matrix.data.ptr,
            cuda_dtype,
            cusv.MatrixLayout.ROW,
            0,  # adjoint flag (handled manually)
            n_targets,
            n_controls,
            compute_type
        )
        
        # ワークスペース確保
        if workspace_size > 0:
            workspace = cp.cuda.memory.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0
        
        # numpy配列に変換（custatevecはホスト側のポインタを期待）
        targets_np = np.array(targets, dtype=np.int32)
        
        if n_controls > 0:
            controls_np = np.array(controls, dtype=np.int32)
            control_values_np = np.ones(n_controls, dtype=np.int32)  # |1⟩で制御
            controls_ptr = controls_np.ctypes.data
            control_values_ptr = control_values_np.ctypes.data
        else:
            controls_ptr = 0
            control_values_ptr = 0
        
        # ゲート適用
        cusv.apply_matrix(
            self.handle,
            self.state.data.ptr,
            cuda_dtype,
            self.n_qubits,
            matrix.data.ptr,
            cuda_dtype,
            cusv.MatrixLayout.ROW,
            0,  # adjoint (we handle it manually above)
            targets_np.ctypes.data,
            n_targets,
            controls_ptr,
            control_values_ptr,
            n_controls,
            compute_type,
            workspace_ptr,
            workspace_size
        )
    
    def apply_cnot(self, control: int, target: int):
        """CNOTゲートを適用"""
        self.apply_gate(self._gate_cache['x'], [target], [control])
    
    def apply_cz(self, control: int, target: int):
        """CZゲートを適用"""
        self.apply_gate(self._gate_cache['z'], [target], [control])
    
    def apply_swap(self, qubit1: int, qubit2: int):
        """SWAPゲートを適用（3つのCNOTに分解）"""
        self.apply_cnot(qubit1, qubit2)
        self.apply_cnot(qubit2, qubit1)
        self.apply_cnot(qubit1, qubit2)
    
    # ==================== 期待値計算 ====================
    
    def expectation_pauli(self, pauli_ops: List[Tuple[str, int]]) -> float:
        """
        Pauli演算子の期待値を計算
        
        Args:
            pauli_ops: [(pauli_type, qubit_index), ...]
                       例: [('Z', 0), ('X', 1)] = Z₀ ⊗ X₁
        
        Returns:
            期待値（実数）
        """
        if not pauli_ops:
            # Identity
            return 1.0
        
        # Pauli演算子を状態に適用
        result_state = self.state.copy()
        
        for pauli_type, qubit in pauli_ops:
            matrix = self._gate_cache[pauli_type.lower()]
            
            # 一時的なStateVectorを使わず直接適用
            # ハンドル再利用のため、同じハンドルを使う一時オブジェクトを作成
            sv_temp = StateVector(self.n_qubits, self.dtype, handle=self.handle)
            sv_temp.state = result_state
            sv_temp.apply_gate(matrix, [qubit])
            result_state = sv_temp.state
        
        # <ψ|P|ψ> = <ψ|result>
        expectation = cp.vdot(self.state, result_state)
        return float(expectation.real.get())
    
    def expectation_z(self, qubit: int) -> float:
        """Z期待値を計算（最適化版）"""
        probs = cp.abs(self.state) ** 2
        mask = cp.arange(self.dim) & (1 << qubit)
        exp_val = cp.sum(probs[mask == 0]) - cp.sum(probs[mask != 0])
        return float(exp_val.get())
    
    # ==================== サンプリング ====================
    
    def sample(self, n_shots: int) -> Dict[str, int]:
        """
        状態をサンプリング
        
        Args:
            n_shots: ショット数
        
        Returns:
            カウント辞書 {"00": 50, "11": 50, ...}
        """
        probs = cp.abs(self.state) ** 2
        probs_np = probs.get()
        
        # サンプリング
        indices = np.random.choice(self.dim, size=n_shots, p=probs_np)
        
        # カウント
        counts = {}
        for idx in indices:
            bitstring = format(idx, f'0{self.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts
    
    def __del__(self):
        """リソース解放"""
        if hasattr(self, '_own_handle') and self._own_handle:
            try:
                cusv.destroy(self.handle)
            except:
                pass


class AdjointDifferentiator:
    """
    Adjoint Differentiation による勾配計算
    
    Forward passで中間状態を保存し、
    Backward passで逆順にadjointゲートを適用しながら勾配を計算
    """
    
    def __init__(self, n_qubits: int, dtype=cp.complex64):
        self.n_qubits = n_qubits
        self.dtype = dtype
        self.handle = cusv.create()
        
    def __del__(self):
        try:
            cusv.destroy(self.handle)
        except:
            pass
    
    def forward_and_gradient(
        self,
        params: np.ndarray,          # [batch_size, n_params]
        operations: List[GateOp],    # 回路操作リスト
        observable: List[Tuple[float, List[Tuple[str, int]]]],  # ハミルトニアン
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward + Backward (Adjoint Method) で期待値と勾配を計算
        
        Args:
            params: パラメータ [batch_size, n_params]
            operations: ゲート操作のリスト
            observable: ハミルトニアン [(coeff, [(pauli, qubit), ...]), ...]
        
        Returns:
            expectations: [batch_size] 期待値
            gradients: [batch_size, n_params] 勾配
        """
        # paramsが1次元の場合は[1, n_params]に変換
        if params.ndim == 1:
            params = params.reshape(1, -1)
            
        batch_size = params.shape[0]
        n_params = params.shape[1]
        
        expectations = np.zeros(batch_size)
        gradients = np.zeros((batch_size, n_params))
        
        for b in range(batch_size):
            exp_val, grad = self._forward_and_gradient_single(
                params[b], operations, observable
            )
            expectations[b] = exp_val
            gradients[b] = grad
        
        return expectations, gradients
    
    def _forward_and_gradient_single(
        self,
        params: np.ndarray,          # [n_params]
        operations: List[GateOp],
        observable: List[Tuple[float, List[Tuple[str, int]]]]
    ) -> Tuple[float, np.ndarray]:
        """単一バッチの forward + gradient"""
        
        n_params = len(params)
        sv = StateVector(self.n_qubits, self.dtype, handle=self.handle)
        
        # ===== Forward Pass =====
        # 各ゲート適用後の状態を保存
        forward_states = [sv.get_state()]
        gate_info = []  # (gate_type, targets, controls, param_idx, param_value)
        
        for op in operations:
            targets = op.targets
            controls = op.controls if op.controls else []
            param_idx = op.param_index
            
            # パラメトリックゲートの場合
            if param_idx is not None and param_idx >= 0:
                param_val = float(params[param_idx])
                matrix = sv.get_gate_matrix(op.gate_type, param_val)
                gate_info.append((op.gate_type, targets, controls, param_idx, param_val))
            else:
                if op.gate_type in [GateType.CNOT, GateType.CZ, GateType.SWAP]:
                    matrix = None
                else:
                    matrix = sv.get_gate_matrix(op.gate_type)
                gate_info.append((op.gate_type, targets, controls, None, None))
            
            # ゲート適用
            if op.gate_type == GateType.CNOT:
                sv.apply_cnot(controls[0], targets[0])
            elif op.gate_type == GateType.CZ:
                sv.apply_cz(controls[0], targets[0])
            elif op.gate_type == GateType.SWAP:
                sv.apply_swap(targets[0], targets[1])
            else:
                sv.apply_gate(matrix, targets, controls)
            
            forward_states.append(sv.get_state())
        
        # ===== 期待値計算 =====
        exp_val = 0.0
        for coeff, pauli_ops in observable:
            exp_val += coeff * sv.expectation_pauli(pauli_ops)
        
        # ===== Backward Pass (Adjoint Method) =====
        # |φ⟩ = O|ψ⟩ から開始
        phi = StateVector(self.n_qubits, self.dtype, handle=self.handle)
        phi_state = cp.zeros_like(sv.state)
        
        for coeff, pauli_ops in observable:
            temp_state = forward_states[-1].copy()
            temp_sv = StateVector(self.n_qubits, self.dtype, handle=self.handle)
            temp_sv.state = temp_state
            
            for pauli_type, qubit in pauli_ops:
                matrix = temp_sv._gate_cache[pauli_type.lower()]
                temp_sv.apply_gate(matrix, [qubit])
            
            phi_state += coeff * temp_sv.state
        
        phi.state = phi_state
        
        # 勾配計算
        grads = np.zeros(n_params)
        
        for i in reversed(range(len(gate_info))):
            gate_type, targets, controls, param_idx, param_val = gate_info[i]
            psi_before = forward_states[i]
            
            if param_idx is not None:
                # パラメトリックゲート: 勾配計算
                # ∂⟨O⟩/∂θ = 2 Re⟨φ|G'|ψ⟩
                deriv_matrix = sv.get_gate_derivative(gate_type, param_val)
                
                # G'|ψ_before⟩ を計算
                psi_deriv_sv = StateVector(self.n_qubits, self.dtype, handle=self.handle)
                psi_deriv_sv.state = psi_before.copy()
                psi_deriv_sv.apply_gate(deriv_matrix, targets, controls)
                
                # ⟨φ|G'|ψ⟩
                inner = cp.vdot(phi.state, psi_deriv_sv.state)
                # 微分の定義によっては係数が変わる可能性があるので注意
                # H = -0.5 Z, dU/dtheta = -i/2 sigma U
                # ここでは行列として直接微分を使っている
                grads[param_idx] += 2.0 * float(inner.real.get())
            
            # U†を|φ⟩に適用
            if gate_type == GateType.CNOT:
                phi.apply_cnot(controls[0], targets[0])
            elif gate_type == GateType.CZ:
                phi.apply_cz(controls[0], targets[0])
            elif gate_type == GateType.SWAP:
                phi.apply_swap(targets[0], targets[1])
            elif param_idx is not None:
                matrix = sv.get_gate_matrix(gate_type, param_val)
                phi.apply_gate(matrix, targets, controls, adjoint=True)
            else:
                matrix = sv.get_gate_matrix(gate_type)
                phi.apply_gate(matrix, targets, controls, adjoint=True)
        
        return exp_val, grads
    
    def forward_only(
        self,
        params: np.ndarray,
        operations: List[GateOp],
        observable: List[Tuple[float, List[Tuple[str, int]]]]
    ) -> np.ndarray:
        """勾配不要の場合のforward only"""
        if params.ndim == 1:
            params = params.reshape(1, -1)
            
        batch_size = params.shape[0]
        expectations = np.zeros(batch_size)
        
        for b in range(batch_size):
            sv = StateVector(self.n_qubits, self.dtype, handle=self.handle)
            
            for op in operations:
                param_idx = op.param_index
                
                if param_idx is not None and param_idx >= 0:
                    param_val = float(params[b, param_idx])
                    matrix = sv.get_gate_matrix(op.gate_type, param_val)
                elif op.gate_type in [GateType.CNOT, GateType.CZ, GateType.SWAP]:
                    matrix = None
                else:
                    matrix = sv.get_gate_matrix(op.gate_type)
                
                if op.gate_type == GateType.CNOT:
                    ctrl = op.controls[0] if op.controls else op.targets[0]
                    tgt = op.targets[-1] if len(op.targets) > 1 else op.targets[0]
                    sv.apply_cnot(ctrl, tgt)
                elif op.gate_type == GateType.CZ:
                    ctrl = op.controls[0] if op.controls else op.targets[0]
                    tgt = op.targets[-1] if len(op.targets) > 1 else op.targets[0]
                    sv.apply_cz(ctrl, tgt)
                elif op.gate_type == GateType.SWAP:
                    sv.apply_swap(op.targets[0], op.targets[1])
                else:
                    sv.apply_gate(matrix, op.targets, op.controls)
            
            # 期待値
            exp_val = 0.0
            for coeff, pauli_ops in observable:
                exp_val += coeff * sv.expectation_pauli(pauli_ops)
            
            expectations[b] = exp_val
        
        return expectations
