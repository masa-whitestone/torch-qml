cmake_minimum_required(VERSION 3.18)
project(torchqml LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# CUDA
# CUDA
# We handle library paths manually in setup.py for pip installs, so Toolkit is optional/quiet
find_package(CUDAToolkit QUIET)

# Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# cuQuantum (cuStateVec)
# 環境変数 CUQUANTUM_ROOT を使用
if(NOT DEFINED CUQUANTUM_ROOT)
    if(DEFINED ENV{CUQUANTUM_ROOT})
        set(CUQUANTUM_ROOT $ENV{CUQUANTUM_ROOT})
    else()
        set(CUQUANTUM_ROOT "/usr/local/cuquantum")
    endif()
endif()

# Python CUDA libs injection to override system
if(DEFINED PYTHON_CUDART_ROOT)
    message(STATUS "Using Python CUDART Root: ${PYTHON_CUDART_ROOT}")
    include_directories(BEFORE SYSTEM ${PYTHON_CUDART_ROOT}/include)
    link_directories(BEFORE ${PYTHON_CUDART_ROOT}/lib)
endif()

if(DEFINED PYTHON_CUBLAS_ROOT)
    message(STATUS "Using Python CUBLAS Root: ${PYTHON_CUBLAS_ROOT}")
    include_directories(BEFORE SYSTEM ${PYTHON_CUBLAS_ROOT}/include)
    link_directories(BEFORE ${PYTHON_CUBLAS_ROOT}/lib)
endif()

if(DEFINED PYTHON_NVCC_ROOT)
    message(STATUS "Using Python NVCC Root: ${PYTHON_NVCC_ROOT}")
    include_directories(BEFORE SYSTEM ${PYTHON_NVCC_ROOT}/include)
endif()

# conda環境やvenv内のincludeを探すためのヒントを追加
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE PYTHON_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(DEFINED CUSTATEVEC_LIB_PATH AND EXISTS "${CUSTATEVEC_LIB_PATH}")
    set(CUSTATEVEC_LIB "${CUSTATEVEC_LIB_PATH}")
else()
    find_library(CUSTATEVEC_LIB custatevec
        HINTS 
            ${CUSTATEVEC_LIBRARY_DIR}
            ${CUQUANTUM_ROOT}/lib 
            ${CUQUANTUM_ROOT}/lib64
            ${PYTHON_PREFIX}/lib
            ${PYTHON_PREFIX}/lib64
    )
endif()

find_path(CUSTATEVEC_INCLUDE custatevec.h
    HINTS 
        ${CUSTATEVEC_INCLUDE_DIR}
        ${CUQUANTUM_ROOT}/include
        ${PYTHON_PREFIX}/include
)

if(NOT CUSTATEVEC_LIB OR NOT CUSTATEVEC_INCLUDE)
    message(FATAL_ERROR "cuStateVec not found. Lib: ${CUSTATEVEC_LIB}, Include: ${CUSTATEVEC_INCLUDE}. Set CUQUANTUM_ROOT or ensure it is installed.")
endif()

message(STATUS "cuStateVec lib: ${CUSTATEVEC_LIB}")
message(STATUS "cuStateVec include: ${CUSTATEVEC_INCLUDE}")

add_subdirectory(csrc)
