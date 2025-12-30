#include "adjoint_diff.h"
#include <cmath>
#include <memory>
#include <map>

namespace torchqml {

// 内部クラス: Forward/Backward計算を担当
class AdjointDifferentiatorImpl {
public:
    AdjointDifferentiatorImpl(int n_qubits) 
        : n_qubits_(n_qubits), sv_(n_qubits) {}
    
    std::pair<float, std::vector<float>> compute(
        const float* params,
        int n_params,
        const std::vector<GateOp>& operations,
        const std::vector<PauliTerm>& observable
    ) {
        // ========== Forward Pass ==========
        sv_.reset();
        
        // 中間状態を保存
        std::vector<std::unique_ptr<StateVector>> forward_states;
        forward_states.push_back(sv_.clone());
        
        // ゲート情報を保存 (param_index, param_value, gate_type, target, controls)
        struct GateInfo {
            int param_index;
            float param_value;
            GateType gate_type;
            int target;
            std::vector<int> controls;
        };
        std::vector<GateInfo> gate_infos;
        
        for (const auto& op : operations) {
            int target = op.targets[0];  // 主ターゲット
            
            float param_value = 0.0f;
            if (op.param_index >= 0) {
                param_value = params[op.param_index];
            }
            
            // ゲート適用
            apply_op(sv_, op, param_value);
            
            // 状態とゲート情報を保存
            forward_states.push_back(sv_.clone());
            gate_infos.push_back({
                op.param_index,
                param_value,
                op.gate_type,
                target,
                op.controls
            });
        }
        
        // ========== 期待値計算 ==========
        float exp_val = compute_expectation(sv_, observable);
        
        // ========== Backward Pass (Adjoint Method) ==========
        std::vector<float> gradients(n_params, 0.0f);
        
        // |φ⟩ = O|ψ⟩ を計算
        StateVector phi(n_qubits_);
        phi.set_state(forward_states.back()->get_state());  // Note: get_state returns host numpy, need proper device copy
        // Optimization: clone from device
        // But for now using the pattern from prototype: 
        // Ideally we should use clone() but the prototype logic used set_state from last forward state which is already device-side if we kept it that way.
        // Actually, let's fix this efficient copy logic.
        // phi is already init.
        // We need to apply observable to phi.
        
        // observable を適用
        apply_observable(phi, observable);
        
        // 逆順にゲートを適用しながら勾配計算
        for (int i = static_cast<int>(gate_infos.size()) - 1; i >= 0; --i) {
            const auto& info = gate_infos[i];
            
            if (info.param_index >= 0) {
                // パラメトリックゲート: 勾配計算
                // G'|ψ_before⟩ を計算
                StateVector psi_deriv(n_qubits_);
                psi_deriv.set_state(forward_states[i]->get_state()); // TODO: Optimize transfer
                
                apply_derivative(psi_deriv, info.gate_type, info.target, 
                                info.param_value, info.controls);
                
                // ⟨φ|G'|ψ⟩
                float inner = compute_inner_product(phi, psi_deriv);
                gradients[info.param_index] += 2.0f * inner;
            }
            
            // U†を|φ⟩に適用
            apply_op_adjoint(phi, info.gate_type, info.target, 
                            info.param_value, info.controls);
        }
        
        return {exp_val, gradients};
    }
    
private:
    int n_qubits_;
    StateVector sv_;
    
    void apply_op(StateVector& sv, const GateOp& op, float param) {
        int target = op.targets[0];
        
        switch (op.gate_type) {
            case GateType::H:
                sv.apply_h(target);
                break;
            case GateType::X:
                if (op.controls.empty()) {
                    sv.apply_x(target);
                } else {
                    sv.apply_cnot(op.controls[0], target);
                }
                break;
            case GateType::Y:
                sv.apply_y(target);
                break;
            case GateType::Z:
                if (op.controls.empty()) {
                    sv.apply_z(target);
                } else {
                    sv.apply_cz(op.controls[0], target);
                }
                break;
            case GateType::S:
                sv.apply_s(target);
                break;
            case GateType::T:
                sv.apply_t(target);
                break;
            case GateType::RX:
                sv.apply_rx(target, param);
                break;
            case GateType::RY:
                sv.apply_ry(target, param);
                break;
            case GateType::RZ:
                sv.apply_rz(target, param);
                break;
            case GateType::CNOT:
                sv.apply_cnot(op.controls.empty() ? op.targets[0] : op.controls[0],
                             op.targets.back());
                break;
            case GateType::CZ:
                sv.apply_cz(op.controls.empty() ? op.targets[0] : op.controls[0],
                           op.targets.back());
                break;
            case GateType::SWAP:
                sv.apply_swap(op.targets[0], op.targets[1]);
                break;
        }
    }
    
    void apply_op_adjoint(StateVector& sv, GateType type, int target,
                         float param, const std::vector<int>& controls) {
        // 自己随伴ゲート (H, X, Y, Z, CNOT, CZ, SWAP) は同じ
        // パラメトリックゲートは -param で適用
        
        switch (type) {
            case GateType::H:
                sv.apply_h(target);  // H† = H
                break;
            case GateType::X:
                if (controls.empty()) {
                    sv.apply_x(target);
                } else {
                    sv.apply_cnot(controls[0], target);
                }
                break;
            case GateType::Y:
                sv.apply_y(target);  // Y† = Y
                break;
            case GateType::Z:
                if (controls.empty()) {
                    sv.apply_z(target);
                } else {
                    sv.apply_cz(controls[0], target);
                }
                break;
            case GateType::S:
                sv.apply_s(target, true);  // S†
                break;
            case GateType::T:
                sv.apply_t(target, true);  // T†
                break;
            case GateType::RX:
                sv.apply_rx(target, -param);  // RX†(θ) = RX(-θ)
                break;
            case GateType::RY:
                sv.apply_ry(target, -param);
                break;
            case GateType::RZ:
                sv.apply_rz(target, -param);
                break;
            case GateType::CNOT:
                sv.apply_cnot(controls.empty() ? target : controls[0], target);
                break;
            case GateType::CZ:
                sv.apply_cz(controls.empty() ? target : controls[0], target);
                break;
            case GateType::SWAP:
                sv.apply_swap(target, target);  // SWAP† = SWAP
                break;
        }
    }
    
    void apply_derivative(StateVector& sv, GateType type, int target,
                         float param, const std::vector<int>& controls) {
        switch (type) {
            case GateType::RX:
                sv.apply_rx_derivative(target, param);
                break;
            case GateType::RY:
                sv.apply_ry_derivative(target, param);
                break;
            case GateType::RZ:
                sv.apply_rz_derivative(target, param);
                break;
            default:
                throw std::runtime_error("Non-parametric gate has no derivative");
        }
    }
    
    void apply_observable(StateVector& sv, const std::vector<PauliTerm>& observable) {
        // 簡易実装: 最初の項のみ適用 (TODO: 複数項の線形結合)
        if (observable.empty()) return;
        
        const auto& term = observable[0];
        for (const auto& [pauli, qubit] : term.paulis) {
            switch (pauli) {
                case 'X': case 'x':
                    sv.apply_x(qubit);
                    break;
                case 'Y': case 'y':
                    sv.apply_y(qubit);
                    break;
                case 'Z': case 'z':
                    sv.apply_z(qubit);
                    break;
            }
        }
    }
    
    float compute_expectation(StateVector& sv, const std::vector<PauliTerm>& observable) {
        float result = 0.0f;
        
        for (const auto& term : observable) {
            float term_exp = sv.expectation_pauli(term.paulis);
            result += term.coefficient * term_exp;
        }
        
        return result;
    }
    
    float compute_inner_product(const StateVector& bra, const StateVector& ket) {
        // <bra|ket> の実部を計算
        // GPU上で計算
        auto result = bra.inner_product(ket);
        return result.real();
    }
};


// ========== 外部インターフェース ==========

std::tuple<py::array_t<float>, py::array_t<float>> adjoint_differentiate(
    py::array_t<float> params,
    const std::vector<GateOp>& operations,
    const std::vector<PauliTerm>& observable,
    int n_qubits
) {
    auto params_buf = params.unchecked<2>();
    int batch_size = params_buf.shape(0);
    int n_params = params_buf.shape(1);
    
    // 結果配列
    py::array_t<float> expectations({batch_size});
    py::array_t<float> gradients({batch_size, n_params});
    
    auto exp_ptr = expectations.mutable_unchecked<1>();
    auto grad_ptr = gradients.mutable_unchecked<2>();
    
    // バッチ処理
    for (int b = 0; b < batch_size; ++b) {
        AdjointDifferentiatorImpl diff(n_qubits);
        
        // パラメータ抽出
        std::vector<float> batch_params(n_params);
        for (int p = 0; p < n_params; ++p) {
            batch_params[p] = params_buf(b, p);
        }
        
        // 計算
        auto [exp_val, grads] = diff.compute(
            batch_params.data(), n_params, operations, observable
        );
        
        // 結果格納
        exp_ptr(b) = exp_val;
        for (int p = 0; p < n_params; ++p) {
            grad_ptr(b, p) = grads[p];
        }
    }
    
    return {expectations, gradients};
}


py::array_t<float> forward_only(
    py::array_t<float> params,
    const std::vector<GateOp>& operations,
    const std::vector<PauliTerm>& observable,
    int n_qubits
) {
    auto params_buf = params.unchecked<2>();
    int batch_size = params_buf.shape(0);
    int n_params = params_buf.shape(1);
    
    py::array_t<float> expectations({batch_size});
    auto exp_ptr = expectations.mutable_unchecked<1>();
    
    for (int b = 0; b < batch_size; ++b) {
        StateVector sv(n_qubits);
        
        for (const auto& op : operations) {
            float param = 0.0f;
            if (op.param_index >= 0) {
                param = params_buf(b, op.param_index);
            }
            
            // ゲート適用 (簡略化)
            int target = op.targets[0];
            switch (op.gate_type) {
                case GateType::H: sv.apply_h(target); break;
                case GateType::X: sv.apply_x(target); break;
                case GateType::Y: sv.apply_y(target); break;
                case GateType::Z: sv.apply_z(target); break;
                case GateType::RX: sv.apply_rx(target, param); break;
                case GateType::RY: sv.apply_ry(target, param); break;
                case GateType::RZ: sv.apply_rz(target, param); break;
                case GateType::CNOT:
                    sv.apply_cnot(op.controls.empty() ? op.targets[0] : op.controls[0],
                                 op.targets.back());
                    break;
                case GateType::CZ:
                    sv.apply_cz(op.controls.empty() ? op.targets[0] : op.controls[0],
                               op.targets.back());
                    break;
                case GateType::SWAP:
                    sv.apply_swap(op.targets[0], op.targets[1]);
                    break;
                default: break;
            }
        }
        
        // 期待値計算
        float exp_val = 0.0f;
        for (const auto& term : observable) {
            exp_val += term.coefficient * sv.expectation_pauli(term.paulis);
        }
        exp_ptr(b) = exp_val;
    }
    
    return expectations;
}

}  // namespace torchqml
