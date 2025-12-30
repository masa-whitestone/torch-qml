#pragma once

#include <custatevec.h>
#include <cuda_runtime.h>
#include <complex>
#include <vector>
#include <memory>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace torchqml {

// エラーチェックマクロ
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +          \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

#define CUSTATEVEC_CHECK(call)                                              \
    do {                                                                    \
        custatevecStatus_t status = call;                                   \
        if (status != CUSTATEVEC_STATUS_SUCCESS) {                          \
            throw std::runtime_error("cuStateVec error: " +                 \
                                     std::to_string(status));               \
        }                                                                   \
    } while (0)


class StateVector {
public:
    /**
     * コンストラクタ
     * @param n_qubits qubit数
     * @param use_double true: complex128, false: complex64
     */
    StateVector(int n_qubits, bool use_double = false);
    
    ~StateVector();
    
    // コピー禁止
    StateVector(const StateVector&) = delete;
    StateVector& operator=(const StateVector&) = delete;
    
    // ムーブ許可
    StateVector(StateVector&&) noexcept;
    StateVector& operator=(StateVector&&) noexcept;
    
    // ========== 状態操作 ==========
    
    /** |0...0⟩ にリセット */
    void reset();
    
    /** 状態をコピー */
    std::unique_ptr<StateVector> clone() const;
    
    /** 状態ベクトル取得 (numpy配列として) */
    py::array_t<std::complex<float>> get_state() const;
    py::array_t<std::complex<double>> get_state_double() const;
    
    /** 状態ベクトル設定 */
    void set_state(py::array_t<std::complex<float>> state);
    void set_state_double(py::array_t<std::complex<double>> state);
    
    /** 生ポインタ取得 (内部用) */
    void* state_ptr() { return d_state_; }
    const void* state_ptr() const { return d_state_; }
    
    // ========== 汎用ゲート適用 ==========
    
    /**
     * 任意のゲート行列を適用
     * @param matrix ゲート行列 [2^k, 2^k] (numpy配列)
     * @param targets ターゲットqubitインデックス
     * @param controls 制御qubitインデックス
     * @param adjoint true: エルミート共役を適用
     */
    void apply_gate(
        py::array_t<std::complex<float>> matrix,
        std::vector<int> targets,
        std::vector<int> controls = {},
        bool adjoint = false
    );
    
    // ========== 特化ゲート (高速) ==========
    
    // 非パラメトリック
    void apply_h(int target);
    void apply_x(int target);
    void apply_y(int target);
    void apply_z(int target);
    void apply_s(int target, bool adjoint = false);
    void apply_t(int target, bool adjoint = false);
    
    // パラメトリック
    void apply_rx(int target, float theta);
    void apply_ry(int target, float theta);
    void apply_rz(int target, float theta);
    
    // パラメトリック微分行列を適用 (Adjoint Diff用)
    void apply_rx_derivative(int target, float theta);
    void apply_ry_derivative(int target, float theta);
    void apply_rz_derivative(int target, float theta);
    
    // 2qubit
    void apply_cnot(int control, int target);
    void apply_cz(int control, int target);
    void apply_swap(int qubit1, int qubit2);
    
    // ========== 期待値計算 ==========
    
    /** Z期待値 (最適化版) */
    float expectation_z(int qubit);
    
    /** Inner product <this|other> */
    std::complex<float> inner_product(const StateVector& other) const;
    
    /** 任意Pauli積の期待値 */
    float expectation_pauli(const std::vector<std::pair<char, int>>& paulis);
    
    // ========== プロパティ ==========
    
    int n_qubits() const { return n_qubits_; }
    int64_t dim() const { return dim_; }
    bool use_double() const { return use_double_; }
    
private:
    int n_qubits_;
    int64_t dim_;
    bool use_double_;
    
    // cuStateVec ハンドル
    custatevecHandle_t handle_;
    
    // GPU上の状態ベクトル
    void* d_state_;  // complex64 or complex128
    
    // ワークスペース
    void* d_workspace_;
    size_t workspace_size_;
    
    // 内部ヘルパー
    void ensure_workspace(size_t required_size);
    void apply_gate_internal(
        const void* matrix,
        int n_targets,
        const int* targets,
        int n_controls,
        const int* controls,
        bool adjoint
    );
    
    // 事前計算されたゲート行列 (GPU上)
    static void* d_gate_h_;
    static void* d_gate_x_;
    static void* d_gate_y_;
    static void* d_gate_z_;
    static void* d_gate_s_;
    static void* d_gate_t_;
    static bool gates_initialized_;
    static void init_static_gates();
};

}  // namespace torchqml
