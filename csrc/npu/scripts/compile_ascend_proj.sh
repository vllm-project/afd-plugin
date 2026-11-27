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
            echo "ERROR: registered operator source directory missing: $subdir" >&2
            return 1
        fi
        if [ -d "$subdir/op_host" ]; then
            cp -rf "$subdir/op_host/"* "$dst_dir/op_host/"
        fi
        if [ -d "$subdir/op_kernel" ]; then
            cp -rf "$subdir/op_kernel/"* "$dst_dir/op_kernel/"
        fi
        # SoC variants with a shared ACLNN ABI use the common op_api.
        local api_dir="$subdir/op_api"
        if [[ ! -d "$api_dir" && "$name" == *_arch35 ]]; then
            api_dir="$src_dir/${name%_arch35}/op_api"
            if [[ ! -d "$api_dir" ]]; then
                echo "ERROR: shared ACLNN sources missing: $api_dir" >&2
                return 1
            fi
        fi
        if [ -d "$api_dir" ]; then
            cp -rf "$api_dir/"* "$dst_dir/pregen/build_out/autogen/"
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
    cp "${src_path}/../../LICENSE" "${build_root}/${proj_name}/LICENSE"

    # Copy the op_host and op_kernel CMakeLists.txt (npu_op_* versions)
    cp "${src_path}/ascend_kernels/cmake_files/op_host/CMakeLists.txt" "${build_root}/${proj_name}/op_host/"
    cp "${src_path}/ascend_kernels/cmake_files/op_kernel/CMakeLists.txt" "${build_root}/${proj_name}/op_kernel/"

    # Copy op_host/op_kernel sources and op_api interface files of the selected
    # operators (include mode)
    copy_ops_include "${src_path}/ascend_kernels" "${build_root}/${proj_name}" "$selected_ops"

    # Stage the four shared Mega protocol/host headers beside the merged
    # operator directories. Third-party headers are resolved from the CANN SDK.
    case " $selected_ops " in
        *" afd_async_atten_send_arch35 "*|*" afd_async_mega_ffn_arch35 "*|*" afd_async_atten_recv_arch35 "*)
            cp -a "${src_path}/ascend_kernels/mega_utils" "${build_root}/${proj_name}/"
            ;;
    esac

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
    local -a generator_args=()
    if [ -d mega_utils ]; then
        # The pinned SDK parser adaptation operates on generated Make recipes.
        generator_args=(-G "Unix Makefiles")
    fi
    cmake -S . -B build_out --preset=default "${generator_args[@]}" -DCMAKE_BUILD_TYPE="$build_type"

    # ### PATCH START: build-local SDK format/OPC adaptation from UMDK 7d25107
    # Preserve SDK validation: only add the FP4 weight format to a local copy.
    if [ -d mega_utils ]; then
        python3 - <<'PYBUILD'
import ast
import re
import shlex
from pathlib import Path

files = list(Path("build_out/op_kernel/CMakeFiles").glob("*ops_info*.dir/build.make"))
pattern = re.compile(r"(?m)^(\t[^\n]*?)(?:\"([^\"\n]*/parse_ini_to_json\.py)\"|'([^'\n]*/parse_ini_to_json\.py)'|(/[^\s\"']*/parse_ini_to_json\.py))")
sources = {m[2] or m[3] or m[4] for make in files for m in pattern.finditer(make.read_text())}
if len(sources) != 1:
    raise RuntimeError(f"Expected one SDK ops-info parser, got {sources}")
source = Path(sources.pop())
text = source.read_text()
formats = re.search(r"(?ms)^FORMAT_LIST\s*=\s*(\[.*?^\])", text)
if formats is None:
    raise RuntimeError(f"FORMAT_LIST definition not found in {source}")
if "FRACTAL_NZ_C0_32" not in ast.literal_eval(formats[1]):
    pos = formats.start(1) + 1
    text = text[:pos] + '\n    "FRACTAL_NZ_C0_32",' + text[pos:]
local = Path("build_out/mega_ops_info/parse_ini_to_json.py").resolve()
local.parent.mkdir(parents=True, exist_ok=True)
local.write_text(text)
for make in files:
    make.write_text(pattern.sub(lambda m: m[1] + shlex.quote(str(local)), make.read_text()))
print(f"Mega ops-info parser: {source} -> {local}")

# Forward the same options as op_kernel/CMakeLists.txt when the SDK's simple
# compile recipe omits them. Do not overwrite a non-empty SDK option string.
options = Path("build_out/mega_kernel_options.txt").read_text().split(";")
compile_options = shlex.quote(" ".join(shlex.quote(option) for option in options))
for make in Path("build_out/op_kernel/CMakeFiles").rglob("build.make"):
    text = make.read_text()
    updated = text.replace('--compile-options=""', "--compile-options=" + compile_options)
    if updated != text:
        make.write_text(updated)
PYBUILD
    fi
    # ### PATCH END: build-local SDK format/OPC adaptation from UMDK 7d25107


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
