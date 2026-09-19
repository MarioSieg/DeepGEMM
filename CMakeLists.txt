# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")

project(deep_gemm LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Force the project's virtualenv Python.

set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python"

        CACHE FILEPATH "Python interpreter" FORCE)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "Python: ${Python_EXECUTABLE}")

execute_process(

        COMMAND "${Python_EXECUTABLE}" -c

        "import torch; print(torch.utils.cmake_prefix_path)"

        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH

        ERROR_VARIABLE TORCH_ERROR

        RESULT_VARIABLE TORCH_RESULT

        OUTPUT_STRIP_TRAILING_WHITESPACE

)

if(NOT TORCH_RESULT EQUAL 0)

    message(FATAL_ERROR

            "Failed to locate PyTorch using ${Python_EXECUTABLE}:\n${TORCH_ERROR}"

    )

endif()

message(STATUS "Torch CMake prefix: ${TORCH_CMAKE_PREFIX_PATH}")

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

find_package(pybind11 REQUIRED)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wno-psabi -Wno-deprecated-declarations")
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")

set(USE_SYSTEM_NVTX on)
set(CUDA_ARCH_LIST "9.0" CACHE STRING "List of CUDA architectures to compile")
set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# NOTE: not using CMAKE_CUDA_STANDARD/CUDA_STANDARD_REQUIRED here, since older
# CMake releases (e.g. 3.22, still common for CLion/system installs) don't know
# the compile flags for the "CUDA20" dialect and fail at generation time. Passing
# the flag straight to nvcc sidesteps that dialect lookup entirely.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

include_directories(deep_gemm/include third-party/deep_jit/include third-party/cutlass/include third-party/cutlass/tools/util/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include/cccl ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

# The main Python API entrance
pybind11_add_module(_C csrc/python_api.cpp)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_python)

# Enable kernel code indexing with CMake-based IDEs
add_library(deep_gemm_indexing_cuda STATIC csrc/indexing/main.cu)
