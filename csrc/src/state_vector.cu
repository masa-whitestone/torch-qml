#include "state_vector.h"
#include "handle_manager.h"
#include <cstring>
#include <stdexcept>
#include <cublas_v2.h>

namespace torchqml {

// 静的メンバ初期化
void* StateVector::d_gate_h_ = nullptr;
void* StateVector::d_gate_x_ = nullptr;
void* StateVector::d_gate_y_ = nullptr;
void* StateVector::d_gate_z_ = nullptr;
void* StateVector::d_gate_s_ = nullptr;
void* StateVector::d_gate_t_ = nullptr;
bool StateVector::gates_initialized_ = false;

// 基本ゲート行列
static const std::complex<float> h_H[] = {
    {0.7071067811865476f, 0}, {0.7071067811865476f, 0},
    {0.7071067811865476f, 0}, {-0.7071067811865476f, 0}
};

static const std::complex<float> h_X[] = {
    {0, 0}, {1, 0},
    {1, 0}, {0, 0}
};

static const std::complex<float> h_Y[] = {
    {0, 0}, {0, -1},
    {0, 1}, {0, 0}
};

static const std::complex<float> h_Z[] = {
    {1, 0}, {0, 0},
    {0, 0}, {-1, 0}
};

static const std::complex<float> h_S[] = {
    {1, 0}, {0, 0},
    {0, 0}, {0, 1}
};

static const float T_PHASE = 0.7071067811865476f;  // 1/sqrt(2)
static const std::complex<float> h_T[] = {
    {1, 0}, {0, 0},
    {0, 0}, {T_PHASE, T_PHASE}
};


void StateVector::init_static_gates() {
    if (gates_initialized_) return;
    
    size_t gate_size = 4 * sizeof(std::complex<float>);
    
    CUDA_CHECK(cudaMalloc(&d_gate_h_, gate_size));
    CUDA_CHECK(cudaMalloc(&d_gate_x_, gate_size));
    CUDA_CHECK(cudaMalloc(&d_gate_y_, gate_size));
    CUDA_CHECK(cudaMalloc(&d_gate_z_, gate_size));
    CUDA_CHECK(cudaMalloc(&d_gate_s_, gate_size));
    CUDA_CHECK(cudaMalloc(&d_gate_t_, gate_size));
    
    CUDA_CHECK(cudaMemcpy(d_gate_h_, h_H, gate_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate_x_, h_X, gate_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate_y_, h_Y, gate_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate_z_, h_Z, gate_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate_s_, h_S, gate_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate_t_, h_T, gate_size, cudaMemcpyHostToDevice));
    
    gates_initialized_ = true;
}


StateVector::StateVector(int n_qubits, bool use_double)
    : n_qubits_(n_qubits),
      dim_(1LL << n_qubits),
      use_double_(use_double),
      d_state_(nullptr),
      d_workspace_(nullptr),
      workspace_size_(0)
{
    init_static_gates();
    
    // cuStateVec ハンドル取得 (HandleManager)
    handle_ = HandleManager::instance().get_custatevec_handle();
    
    // 状態ベクトル確保
    size_t state_size = dim_ * (use_double_ ? sizeof(std::complex<double>) 
                                            : sizeof(std::complex<float>));
    CUDA_CHECK(cudaMalloc(&d_state_, state_size));
    
    // |0...0⟩ に初期化
    reset();
}


StateVector::~StateVector() {
    if (d_state_) cudaFree(d_state_);
    if (d_workspace_) cudaFree(d_workspace_);
    // ハンドルはマネージャ管理なので破棄しない
}


StateVector::StateVector(StateVector&& other) noexcept
    : n_qubits_(other.n_qubits_),
      dim_(other.dim_),
      use_double_(other.use_double_),
      handle_(other.handle_),
      d_state_(other.d_state_),
      d_workspace_(other.d_workspace_),
      workspace_size_(other.workspace_size_)
{
    other.d_state_ = nullptr;
    other.d_workspace_ = nullptr;
}


StateVector& StateVector::operator=(StateVector&& other) noexcept {
    if (this != &other) {
        if (d_state_) cudaFree(d_state_);
        if (d_workspace_) cudaFree(d_workspace_);
        // ハンドルは破棄しない
        
        n_qubits_ = other.n_qubits_;
        dim_ = other.dim_;
        use_double_ = other.use_double_;
        handle_ = other.handle_;
        d_state_ = other.d_state_;
        d_workspace_ = other.d_workspace_;
        workspace_size_ = other.workspace_size_;
        
        other.d_state_ = nullptr;
        other.d_workspace_ = nullptr;
    }
    return *this;
}


// ... (reset, clone, getters, setters skipped as they are fine) ...


// ... (ensure_workspace, apply_gate_internal skipped) ...


// ... (apply_gate methods skipped) ...


// Note: Ensure header inclusion
#include "handle_manager.h"


// ... (implementations of gates) ...


std::complex<float> StateVector::inner_product(const StateVector& other) const {
    if (use_double_) {
        throw std::runtime_error("Double precision inner_product not implemented");
    }
    
    // cuBLAS で内積計算
    cuComplex result;
    cublasHandle_t cublas_handle = HandleManager::instance().get_cublas_handle();
    
    cublasCdotc(
        cublas_handle,
        dim_,
        static_cast<const cuComplex*>(d_state_), 1,
        static_cast<const cuComplex*>(other.d_state_), 1,
        &result
    );
    
    return std::complex<float>(cuCrealf(result), cuCimagf(result));
}


float StateVector::expectation_pauli(const std::vector<std::pair<char, int>>& paulis) {
    if (paulis.empty()) {
        return 1.0f;  // Identity
    }
    
    // Pauli演算子を状態に適用してから内積を計算
    auto temp = clone();
    
    for (const auto& [pauli_type, qubit] : paulis) {
        switch (pauli_type) {
            case 'X': case 'x':
                temp->apply_x(qubit);
                break;
            case 'Y': case 'y':
                temp->apply_y(qubit);
                break;
            case 'Z': case 'z':
                temp->apply_z(qubit);
                break;
            case 'I': case 'i':
                // Identity: 何もしない
                break;
            default:
                throw std::runtime_error("Unknown Pauli type");
        }
    }
    
    // <ψ|P|ψ> = <ψ|temp>
    // cuBLAS で内積計算
    cuComplex result;
    cublasHandle_t cublas_handle = HandleManager::instance().get_cublas_handle();
    
    cublasCdotc(
        cublas_handle,
        dim_,
        static_cast<cuComplex*>(d_state_), 1,
        static_cast<cuComplex*>(temp->d_state_), 1,
        &result
    );
    
    return cuCrealf(result);  // 実部のみ
}


void StateVector::reset() {
    size_t state_size = dim_ * (use_double_ ? sizeof(std::complex<double>)
                                            : sizeof(std::complex<float>));
    CUDA_CHECK(cudaMemset(d_state_, 0, state_size));
    
    // |0⟩ = [1, 0, 0, ...]
    if (use_double_) {
        std::complex<double> one(1.0, 0.0);
        CUDA_CHECK(cudaMemcpy(d_state_, &one, sizeof(one), cudaMemcpyHostToDevice));
    } else {
        std::complex<float> one(1.0f, 0.0f);
        CUDA_CHECK(cudaMemcpy(d_state_, &one, sizeof(one), cudaMemcpyHostToDevice));
    }
}


std::unique_ptr<StateVector> StateVector::clone() const {
    auto copy = std::make_unique<StateVector>(n_qubits_, use_double_);
    size_t state_size = dim_ * (use_double_ ? sizeof(std::complex<double>)
                                            : sizeof(std::complex<float>));
    CUDA_CHECK(cudaMemcpy(copy->d_state_, d_state_, state_size, 
                          cudaMemcpyDeviceToDevice));
    return copy;
}


py::array_t<std::complex<float>> StateVector::get_state() const {
    if (use_double_) {
        throw std::runtime_error("Use get_state_double() for double precision");
    }
    
    py::array_t<std::complex<float>> result(dim_);
    auto buf = result.mutable_data();
    CUDA_CHECK(cudaMemcpy(buf, d_state_, dim_ * sizeof(std::complex<float>),
                          cudaMemcpyDeviceToHost));
    return result;
}


py::array_t<std::complex<double>> StateVector::get_state_double() const {
    if (!use_double_) {
        throw std::runtime_error("Use get_state() for single precision");
    }
    
    py::array_t<std::complex<double>> result(dim_);
    auto buf = result.mutable_data();
    CUDA_CHECK(cudaMemcpy(buf, d_state_, dim_ * sizeof(std::complex<double>),
                          cudaMemcpyDeviceToHost));
    return result;
}


void StateVector::set_state(py::array_t<std::complex<float>> state) {
    if (use_double_) {
        throw std::runtime_error("Use set_state_double() for double precision");
    }
    if (state.size() != dim_) {
        throw std::runtime_error("State size mismatch");
    }
    
    CUDA_CHECK(cudaMemcpy(d_state_, state.data(), dim_ * sizeof(std::complex<float>),
                          cudaMemcpyHostToDevice));
}


void StateVector::set_state_double(py::array_t<std::complex<double>> state) {
    if (!use_double_) {
        throw std::runtime_error("Use set_state() for single precision");
    }
    if (state.size() != dim_) {
        throw std::runtime_error("State size mismatch");
    }
    
    CUDA_CHECK(cudaMemcpy(d_state_, state.data(), dim_ * sizeof(std::complex<double>),
                          cudaMemcpyHostToDevice));
}


void StateVector::ensure_workspace(size_t required_size) {
    if (required_size > workspace_size_) {
        if (d_workspace_) cudaFree(d_workspace_);
        CUDA_CHECK(cudaMalloc(&d_workspace_, required_size));
        workspace_size_ = required_size;
    }
}


void StateVector::apply_gate_internal(
    const void* matrix,
    int n_targets,
    const int* targets,
    int n_controls,
    const int* controls,
    bool adjoint
) {
    cudaDataType_t data_type = use_double_ ? CUDA_C_64F : CUDA_C_32F;
    custatevecComputeType_t compute_type = use_double_ 
        ? CUSTATEVEC_COMPUTE_64F : CUSTATEVEC_COMPUTE_32F;
    
    // ワークスペースサイズ取得
    size_t required_size;
    CUSTATEVEC_CHECK(custatevecApplyMatrixGetWorkspaceSize(
        handle_,
        data_type,
        n_qubits_,
        matrix,
        data_type,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        adjoint ? 1 : 0,
        n_targets,
        n_controls,
        compute_type,
        &required_size
    ));
    
    ensure_workspace(required_size);
    
    // ゲート適用
    CUSTATEVEC_CHECK(custatevecApplyMatrix(
        handle_,
        d_state_,
        data_type,
        n_qubits_,
        matrix,
        data_type,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        adjoint ? 1 : 0,
        targets,
        n_targets,
        controls,
        nullptr,  // control bit values (all 1s by default)
        n_controls,
        compute_type,
        d_workspace_,
        workspace_size_
    ));
}


void StateVector::apply_gate(
    py::array_t<std::complex<float>> matrix,
    std::vector<int> targets,
    std::vector<int> controls,
    bool adjoint
) {
    // 行列をGPUにコピー
    size_t n_targets = targets.size();
    size_t matrix_dim = 1ULL << n_targets;
    size_t matrix_size = matrix_dim * matrix_dim * sizeof(std::complex<float>);
    
    void* d_matrix;
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix_size));
    CUDA_CHECK(cudaMemcpy(d_matrix, matrix.data(), matrix_size, 
                          cudaMemcpyHostToDevice));
    
    apply_gate_internal(
        d_matrix,
        n_targets,
        targets.data(),
        controls.size(),
        controls.empty() ? nullptr : controls.data(),
        adjoint
    );
    
    cudaFree(d_matrix);
}


// ========== 特化ゲート実装 ==========

void StateVector::apply_h(int target) {
    apply_gate_internal(d_gate_h_, 1, &target, 0, nullptr, false);
}

void StateVector::apply_x(int target) {
    apply_gate_internal(d_gate_x_, 1, &target, 0, nullptr, false);
}

void StateVector::apply_y(int target) {
    apply_gate_internal(d_gate_y_, 1, &target, 0, nullptr, false);
}

void StateVector::apply_z(int target) {
    apply_gate_internal(d_gate_z_, 1, &target, 0, nullptr, false);
}

void StateVector::apply_s(int target, bool adjoint) {
    apply_gate_internal(d_gate_s_, 1, &target, 0, nullptr, adjoint);
}

void StateVector::apply_t(int target, bool adjoint) {
    apply_gate_internal(d_gate_t_, 1, &target, 0, nullptr, adjoint);
}


void StateVector::apply_rx(int target, float theta) {
    float c = std::cos(theta / 2);
    float s = std::sin(theta / 2);
    
    std::complex<float> h_rx[] = {
        {c, 0}, {0, -s},
        {0, -s}, {c, 0}
    };
    
    void* d_rx;
    CUDA_CHECK(cudaMalloc(&d_rx, 4 * sizeof(std::complex<float>)));
    CUDA_CHECK(cudaMemcpy(d_rx, h_rx, 4 * sizeof(std::complex<float>),
                          cudaMemcpyHostToDevice));
    
    apply_gate_internal(d_rx, 1, &target, 0, nullptr, false);
    
    cudaFree(d_rx);
}


void StateVector::apply_ry(int target, float theta) {
    float c = std::cos(theta / 2);
    float s = std::sin(theta / 2);
    
    std::complex<float> h_ry[] = {
        {c, 0}, {-s, 0},
        {s, 0}, {c, 0}
    };
    
    void* d_ry;
    CUDA_CHECK(cudaMalloc(&d_ry, 4 * sizeof(std::complex<float>)));
    CUDA_CHECK(cudaMemcpy(d_ry, h_ry, 4 * sizeof(std::complex<float>),
                          cudaMemcpyHostToDevice));
    
    apply_gate_internal(d_ry, 1, &target, 0, nullptr, false);
    
    cudaFree(d_ry);
}


void StateVector::apply_rz(int target, float theta) {
    std::complex<float> e_neg(std::cos(-theta/2), std::sin(-theta/2));
    std::complex<float> e_pos(std::cos(theta/2), std::sin(theta/2));
    
    std::complex<float> h_rz[] = {
        e_neg, {0, 0},
        {0, 0}, e_pos
    };
    
    void* d_rz;
    CUDA_CHECK(cudaMalloc(&d_rz, 4 * sizeof(std::complex<float>)));
    CUDA_CHECK(cudaMemcpy(d_rz, h_rz, 4 * sizeof(std::complex<float>),
                          cudaMemcpyHostToDevice));
    
    apply_gate_internal(d_rz, 1, &target, 0, nullptr, false);
    
    cudaFree(d_rz);
}


void StateVector::apply_rx_derivative(int target, float theta) {
    float c = std::cos(theta / 2);
    float s = std::sin(theta / 2);
    
    // d/dθ RX = [[-s/2, -ic/2], [-ic/2, -s/2]]
    std::complex<float> h_drx[] = {
        {-s/2, 0}, {0, -c/2},
        {0, -c/2}, {-s/2, 0}
    };
    
    void* d_drx;
    CUDA_CHECK(cudaMalloc(&d_drx, 4 * sizeof(std::complex<float>)));
    CUDA_CHECK(cudaMemcpy(d_drx, h_drx, 4 * sizeof(std::complex<float>),
                          cudaMemcpyHostToDevice));
    
    apply_gate_internal(d_drx, 1, &target, 0, nullptr, false);
    
    cudaFree(d_drx);
}


void StateVector::apply_ry_derivative(int target, float theta) {
    float c = std::cos(theta / 2);
    float s = std::sin(theta / 2);
    
    // d/dθ RY = [[-s/2, -c/2], [c/2, -s/2]]
    std::complex<float> h_dry[] = {
        {-s/2, 0}, {-c/2, 0},
        {c/2, 0}, {-s/2, 0}
    };
    
    void* d_dry;
    CUDA_CHECK(cudaMalloc(&d_dry, 4 * sizeof(std::complex<float>)));
    CUDA_CHECK(cudaMemcpy(d_dry, h_dry, 4 * sizeof(std::complex<float>),
                          cudaMemcpyHostToDevice));
    
    apply_gate_internal(d_dry, 1, &target, 0, nullptr, false);
    
    cudaFree(d_dry);
}


void StateVector::apply_rz_derivative(int target, float theta) {
    std::complex<float> de_neg(-0.5f * std::sin(-theta/2), 0.5f * std::cos(-theta/2));
    std::complex<float> de_pos(0.5f * std::sin(theta/2), 0.5f * std::cos(theta/2));
    // d/dθ e^{iθ/2} = i/2 * e^{iθ/2} = i/2 * (cos + i sin) = (-sin/2, cos/2)
    
    std::complex<float> h_drz[] = {
        de_neg, {0, 0},
        {0, 0}, de_pos
    };
    
    void* d_drz;
    CUDA_CHECK(cudaMalloc(&d_drz, 4 * sizeof(std::complex<float>)));
    CUDA_CHECK(cudaMemcpy(d_drz, h_drz, 4 * sizeof(std::complex<float>),
                          cudaMemcpyHostToDevice));
    
    apply_gate_internal(d_drz, 1, &target, 0, nullptr, false);
    
    cudaFree(d_drz);
}


void StateVector::apply_cnot(int control, int target) {
    // CNOTはX行列を制御付きで適用
    if (gates_initialized_) {
        apply_gate_internal(d_gate_x_, 1, &target, 1, &control, false);
    }
}


void StateVector::apply_cz(int control, int target) {
    if (gates_initialized_) {
        apply_gate_internal(d_gate_z_, 1, &target, 1, &control, false);
    }
}


void StateVector::apply_swap(int qubit1, int qubit2) {
    // SWAP = CNOT(1,2) CNOT(2,1) CNOT(1,2)
    apply_cnot(qubit1, qubit2);
    apply_cnot(qubit2, qubit1);
    apply_cnot(qubit1, qubit2);
}


// ========== 期待値計算 ==========

// Z期待値計算カーネル
float StateVector::expectation_z(int qubit) {
    if (use_double_) {
        throw std::runtime_error("Double precision expectation not implemented");
    }
    
    // Use expectation_pauli implementation which is verified to work
    std::vector<std::pair<char, int>> paulis = {{'Z', qubit}};
    return expectation_pauli(paulis);
}



}  // namespace torchqml
