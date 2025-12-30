set(TORCHQML_SOURCES
    src/state_vector.cu
    src/adjoint_diff.cu
    src/handle_manager.cu
    # src/gate_ops.cu  # Merged into state_vector.cu
    # src/pauli_ops.cu # Merged into state_vector.cu
    bindings.cpp
)

# ファイルが存在しないとエラーになるので、まずは作成されるまでbindings.cppのみにするか、touchしておく
# ここではbindings.cppのみでビルド確認するため、他はコメントアウトのままでもよいが
# 後で実装するファイルのためにプレースホルダーファイル作成コマンドを走らせる

pybind11_add_module(_C ${TORCHQML_SOURCES})

target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUSTATEVEC_INCLUDE}
)

if(DEFINED PYTHON_CUDART_ROOT)
    set(LIB_CUDART "${PYTHON_CUDART_ROOT}/lib/libcudart.so.12")
else()
    set(LIB_CUDART "cudart")
endif()

if(DEFINED PYTHON_CUBLAS_ROOT)
    set(LIB_CUBLAS "${PYTHON_CUBLAS_ROOT}/lib/libcublas.so.12")
else()
    set(LIB_CUBLAS "cublas")
endif()

target_link_libraries(_C PRIVATE
    ${CUSTATEVEC_LIB}
    ${LIB_CUDART}
    ${LIB_CUBLAS}
)

# CUDA アーキテクチャ (top-level config inherits)
set_target_properties(_C PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# インストール先
install(TARGETS _C DESTINATION torchqml)
