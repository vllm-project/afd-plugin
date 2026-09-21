cmake_minimum_required(VERSION 3.16)
project(afd_plugin_ascend_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" "-c" "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_PREFIX_RESULT)
if(NOT TORCH_PREFIX_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to locate torch CMake prefix")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

find_package(Torch REQUIRED)

if(NOT DEFINED TORCH_NPU_PATH OR TORCH_NPU_PATH STREQUAL "")
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c" "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_NPU_RESULT)
  if(NOT TORCH_NPU_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to locate torch_npu")
  endif()
endif()

if(NOT DEFINED ASCEND_HOME_PATH OR ASCEND_HOME_PATH STREQUAL "")
  set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(ASCEND_HOME_PATH STREQUAL "")
  set(ASCEND_HOME_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
endif()
if(ASCEND_HOME_PATH STREQUAL "")
  set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(AFD_PLUGIN_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(AFD_NPU_CSRC_DIR "${AFD_PLUGIN_ROOT}/csrc/npu")
set(AFD_CANN_OPS_DIR "${AFD_PLUGIN_ROOT}/afd_plugin/_cann_ops_custom/vendors/afd-plugin/op_api/lib")

pybind11_add_module(afd_plugin_ascend_C
  "${CMAKE_CURRENT_LIST_DIR}/torch_binding.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/torch_binding_meta.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/torch_binding_gmm_swiglu_quant.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/pytorch_extension/NPUBridge.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/pytorch_extension/NPUStorageImpl.cpp")

# Keep registration aligned with the run package selected by setup.py. The
# asynchronous CAM kernels use the A3 HCCL resource ABI and do not support A5.
if(NOT DEFINED AFD_SOC_VERSION OR AFD_SOC_VERSION STREQUAL "")
  set(AFD_SOC_VERSION "$ENV{SOC_VERSION}")
endif()
if(AFD_SOC_VERSION STREQUAL "")
  set(AFD_SOC_VERSION "910c")
endif()
if(AFD_SOC_VERSION STREQUAL "910c" OR AFD_SOC_VERSION MATCHES "^ascend910_93")
  target_sources(afd_plugin_ascend_C PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/torch_binding_cam_async.cpp")
elseif(AFD_SOC_VERSION STREQUAL "950" OR AFD_SOC_VERSION MATCHES "^[Aa]scend950")
  message(STATUS "CAM asynchronous bindings disabled for Ascend 950")
else()
  message(FATAL_ERROR "Unsupported AFD_SOC_VERSION: ${AFD_SOC_VERSION}")
endif()

set_target_properties(afd_plugin_ascend_C PROPERTIES OUTPUT_NAME "_C_ascend")
target_compile_definitions(afd_plugin_ascend_C PRIVATE TORCH_EXTENSION_NAME=_C_ascend)

target_include_directories(afd_plugin_ascend_C PRIVATE
  "${AFD_NPU_CSRC_DIR}"
  "${AFD_NPU_CSRC_DIR}/pybind/pytorch_extension"
  "${AFD_NPU_CSRC_DIR}/ascend_kernels"
  "${TORCH_INCLUDE_DIRS}"
  "${TORCH_NPU_PATH}/include"
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/aclnn"
  "${ASCEND_HOME_PATH}/include/aclnn_kernels")

target_link_directories(afd_plugin_ascend_C PRIVATE
  "${TORCH_LIBRARY_DIRS}"
  "${TORCH_NPU_PATH}/lib"
  "${ASCEND_HOME_PATH}/lib64"
  "${AFD_CANN_OPS_DIR}")

target_link_libraries(afd_plugin_ascend_C PRIVATE
  "${TORCH_LIBRARIES}"
  torch_npu
  ascendcl
  dl
  opapi)

target_link_options(afd_plugin_ascend_C PRIVATE
  "-Wl,-rpath,$ORIGIN:$ORIGIN/_cann_ops_custom/vendors/afd-plugin/op_api/lib")

install(TARGETS afd_plugin_ascend_C LIBRARY DESTINATION afd_plugin)
