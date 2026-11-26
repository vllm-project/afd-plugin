# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
cmake_minimum_required(VERSION 3.24)
project(afd_gpu_async LANGUAGES CXX CUDA)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../afd_plugin/connectors/gpu/async_nvshmem/library.py"
    "${NVSHMEM_HOME}"
  OUTPUT_VARIABLE NVSHMEM_HOST OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
get_filename_component(NVSHMEM_LIBDIR "${NVSHMEM_HOST}" DIRECTORY)
get_filename_component(NVSHMEM_ROOT "${NVSHMEM_LIBDIR}" DIRECTORY)
find_library(NVSHMEM_DEVICE_LIBRARY nvshmem_device PATHS "${NVSHMEM_LIBDIR}"
  NO_DEFAULT_PATH NO_CACHE REQUIRED)
pybind11_add_module(_C_gpu_async MODULE async_transport.cu)
target_include_directories(_C_gpu_async PRIVATE "${NVSHMEM_ROOT}/include")
target_link_libraries(_C_gpu_async PRIVATE
  "${NVSHMEM_DEVICE_LIBRARY}" "${NVSHMEM_HOST}" CUDA::cudart CUDA::cuda_driver)
set_target_properties(_C_gpu_async PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_STANDARD 17 CXX_STANDARD 17
  INSTALL_RPATH "${NVSHMEM_LIBDIR}")
install(TARGETS _C_gpu_async LIBRARY DESTINATION afd_plugin)
