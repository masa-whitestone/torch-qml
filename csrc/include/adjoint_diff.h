#pragma once

#include "state_vector.h"
#include <vector>
#include <tuple>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace torchqml {

// ゲートタイプ
enum class GateType {
    H, X, Y, Z, S, T,
    RX, RY, RZ,
    CNOT, CZ, SWAP
};

// ゲート操作
struct GateOp {
    GateType gate_type;
    std::vector<int> targets;
    std::vector<int> controls;
    int param_index;  // -1 if non-parametric
    
    GateOp() : param_index(-1) {}
};

// Pauli項
struct PauliTerm {
    float coefficient;
    std::vector<std::pair<char, int>> paulis;  // [(pauli_type, qubit), ...]
};

/**
 * Adjoint Differentiation
 * 
 * Forward + Backward を一括で計算
 * @param params パラメータ [batch_size, n_params]
 * @param operations ゲート操作リスト
 * @param observable ハミルトニアン (Pauli項のリスト)
 * @param n_qubits qubit数
 * @return (expectations [batch_size], gradients [batch_size, n_params])
 */
std::tuple<py::array_t<float>, py::array_t<float>> adjoint_differentiate(
    py::array_t<float> params,
    const std::vector<GateOp>& operations,
    const std::vector<PauliTerm>& observable,
    int n_qubits
);

/**
 * Forward only (勾配不要時)
 */
py::array_t<float> forward_only(
    py::array_t<float> params,
    const std::vector<GateOp>& operations,
    const std::vector<PauliTerm>& observable,
    int n_qubits
);

}  // namespace torchqml
