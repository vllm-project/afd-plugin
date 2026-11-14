# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
cmake_minimum_required(VERSION 3.24)
project(afd_gpu_async LANGUAGES CXX CUDA)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(NOT NVSHMEM_HOME)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "import importlib.util; print(next(iter(importlib.util.find_spec('nvidia.nvshmem').submodule_search_locations)))"
    OUTPUT_VARIABLE NVSHMEM_HOME OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
endif()
find_library(NVSHMEM_HOST NAMES nvshmem_host libnvshmem_host.so.3
  HINTS "${NVSHMEM_HOME}/lib" REQUIRED)
find_library(NVSHMEM_DEVICE nvshmem_device HINTS "${NVSHMEM_HOME}/lib" REQUIRED)
pybind11_add_module(_C_gpu_async MODULE async_transport.cu)
target_include_directories(_C_gpu_async PRIVATE "${NVSHMEM_HOME}/include")
target_link_libraries(_C_gpu_async PRIVATE
  "${NVSHMEM_DEVICE}" "${NVSHMEM_HOST}" CUDA::cudart CUDA::cuda_driver)
set_target_properties(_C_gpu_async PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_STANDARD 17 CXX_STANDARD 17
  INSTALL_RPATH "${NVSHMEM_HOME}/lib")
install(TARGETS _C_gpu_async LIBRARY DESTINATION afd_plugin)
