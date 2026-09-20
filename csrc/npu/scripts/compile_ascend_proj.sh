#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# AFD Ascend custom-op run package build driver (msopgen + npu_op_* build
# system). Invoked by csrc/npu/build_aclnn.sh; builds the operator project for
# one SOC generation and copies the .run installer to the output directory.
#
# Usage: compile_ascend_proj.sh <src_path> <soc_version> <build_type> [out_dir]
#   src_path    csrc/npu directory (contains ascend_kernels/)
#   soc_version ascend910_93 | ascend950
#   build_type  Release | Debug
#   out_dir     where the .run is placed (default: <src_path>/output)

set -e

# Operator registry path (operator identity = directory name under ascend_kernels/)
REGISTRY_PATH="${1}/ascend_kernels/operator_registry.json"
# select_ops.py path
SELECT_OPS_SCRIPT="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/select_ops.py"

# Whether SHMEM is installed, decided by the SHMEM_HOME_PATH environment
# variable. No AFD operator currently requires it.
shmem_installed=1
if [ -z "${SHMEM_HOME_PATH}" ]; then
    shmem_installed=0
fi

# Copy op_host/op_kernel/op_api of the selected operator directories into the
# target project. utils is the shared header directory and is always copied;
# the remaining entries come from the ops array (operator directory names).
copy_ops_include() {
    local src_dir="$1" # source directory (ascend_kernels)
    local dst_dir="$2" # target directory (project root)
    local -a ops       # space-separated list of operator directory names
    read -r -a ops <<< "$3"

    # Make sure op_host, op_kernel and the pregen autogen directory exist
    mkdir -p "$dst_dir/op_host" "$dst_dir/op_kernel" "$dst_dir/pregen/build_out/autogen"

    # Always copy the shared header directory utils (.h under op_host/op_kernel)
    if [ -d "$src_dir/utils/op_host" ]; then
        cp -rf "$src_dir/utils/op_host/"* "$dst_dir/op_host/" 2>/dev/null || true
    fi
    if [ -d "$src_dir/utils/op_kernel" ]; then
        cp -rf "$src_dir/utils/op_kernel/"* "$dst_dir/op_kernel/" 2>/dev/null || true
    fi

    # Copy op_host/op_kernel/op_api of every selected operator
    for name in "${ops[@]}"; do
        local subdir="$src_dir/$name"
        if [ ! -d "$subdir" ]; then
            echo "Warning: operator dir not found, skipping: $name"
            continue
        fi
        if [ -d "$subdir/op_host" ]; then
            cp -rf "$subdir/op_host/"* "$dst_dir/op_host/"
        fi
        if [ -d "$subdir/op_kernel" ]; then
            cp -rf "$subdir/op_kernel/"* "$dst_dir/op_kernel/"
        fi
        # op_api interface files go to pregen/build_out/autogen
        if [ -d "$subdir/op_api" ]; then
            cp -rf "$subdir/op_api/"* "$dst_dir/pregen/build_out/autogen/"
        fi
    done
}

# Ask select_ops.py for the final operator list; returns space-separated
# operator directory names. On failure (registry/validation error) select_ops.py
# has already printed the error and exited non-zero, so set -e aborts.
resolve_ops() {
    local soc=$1
    python3 "$SELECT_OPS_SCRIPT" \
        --registry "$REGISTRY_PATH" \
        --soc "$soc" \
        --shmem "$shmem_installed" | tr '\n' ' ' | sed 's/  */ /g; s/^ //; s/ $//'
}

# Build the operator project and place its artifacts at the given location
build_ascend_proj() {
    local src_path=$1
    local soc_version=$2
    local build_type=$3
    local out_dir=$4
    local build_root="${src_path}/build"

    mkdir -p "${build_root}" "${out_dir}"

    local proj_name="ascend_kernels_${soc_version}_proj"
    if [ -d "${build_root:?}/${proj_name}" ]; then
        rm -rf "${build_root:?}/${proj_name}"
    fi

    # Resolve the operator list to compile for this run (include mode;
    # select_ops.py validates and filters it)
    local selected_ops
    selected_ops=$(resolve_ops "$soc_version")
    if [ -z "$selected_ops" ]; then
        echo "ERROR: no operators resolved for SOC ${soc_version}"
        return 1
    fi
    echo "Selected operators for ${soc_version}: ${selected_ops}"

    # Use msopgen to generate the per-generation operator project and its
    # CMakePresets.json
    export OPS_PROJECT_NAME=aclnnInner

    echo "msopgen gen -i ${src_path}/ascend_kernels/AddCustom.json -c ai_core-${soc_version} -f pytorch -lan cpp -out ${build_root}/${proj_name}"
    msopgen gen -i "${src_path}/ascend_kernels/AddCustom.json" -c "ai_core-${soc_version}" -f pytorch -lan cpp -out "${build_root}/${proj_name}"
    rm -rf "${build_root}/${proj_name}/op_host/add_custom*"
    rm -rf "${build_root}/${proj_name}/op_kernel/add_custom*"

    # Copy the top-level CMakeLists.txt (CMakePresets.json comes from msopgen)
    cp "${src_path}/ascend_kernels/CMakeLists.txt" "${build_root}/${proj_name}/"

    # Copy the op_host and op_kernel CMakeLists.txt (npu_op_* versions)
    cp "${src_path}/ascend_kernels/cmake_files/op_host/CMakeLists.txt" "${build_root}/${proj_name}/op_host/"
    cp "${src_path}/ascend_kernels/cmake_files/op_kernel/CMakeLists.txt" "${build_root}/${proj_name}/op_kernel/"

    # Copy op_host/op_kernel sources and op_api interface files of the selected
    # operators (include mode)
    copy_ops_include "${src_path}/ascend_kernels" "${build_root}/${proj_name}" "$selected_ops"

    # Copy cmake_files/cmake (custom build functions replacing the msopgen defaults)
    cp -rf "${src_path}/ascend_kernels/cmake_files/cmake" "${build_root}/${proj_name}/"

    # Set build_type/vendor_name in CMakePresets.json (after msopgen has run)
    python3 "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/set_conf.py" \
        "${build_root}/${proj_name}/CMakePresets.json" "$build_type" True afd-plugin

    # CANN package path: try ASCEND_HOME_PATH / ASCEND_CANN_PACKAGE_PATH, then
    # derive from ASCEND_TOOLKIT_HOME, else fall back to the default location.
    if [ -z "${ASCEND_CANN_PACKAGE_PATH}" ]; then
        if [ -n "${ASCEND_HOME_PATH}" ]; then
            export ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
        elif [ -n "${ASCEND_TOOLKIT_HOME}" ]; then
            export ASCEND_CANN_PACKAGE_PATH="${ASCEND_TOOLKIT_HOME}"
        else
            export ASCEND_CANN_PACKAGE_PATH="/usr/local/Ascend/ascend-toolkit/latest"
        fi
    fi

    cd "${build_root}/${proj_name}"

    # Configure cmake
    cmake -S . -B build_out --preset=default -DCMAKE_BUILD_TYPE="$build_type"

    # Build parallelism: configurable through AFD_BUILD_JOBS, default 8
    BUILD_JOBS=${AFD_BUILD_JOBS:-8}
    cmake --build build_out --target binary -j"${BUILD_JOBS}"
    cmake --build build_out --target package -j"${BUILD_JOBS}"

    # Copy out the run package under a fixed name for build_aclnn.sh to pick up
    cp build_out/*.run "${out_dir}/AFD_${soc_version}.run"
    echo "Run package: ${out_dir}/AFD_${soc_version}.run"
}

# ---- entry point ----
src_path=$1
soc_arg=$2
build_type=${3:-Release}
out_dir=${4:-${src_path}/output}

if [ -z "$src_path" ] || [ -z "$soc_arg" ]; then
    echo "Usage: $0 <src_path> <soc_version> <build_type> [out_dir]"
    exit 1
fi

if [ ! -f "$REGISTRY_PATH" ]; then
    echo "ERROR: registry not found: $REGISTRY_PATH"
    exit 1
fi

build_ascend_proj "$src_path" "$soc_arg" "$build_type" "$out_dir"
