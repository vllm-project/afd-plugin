#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Existing-evidence collection only. No restart, signals, or runtime imports.
set -uo pipefail
: "${RUN_ROOT:?先设置 RUN_ROOT 为失败 C5 的日志目录}"
[[ -d "$RUN_ROOT" ]] || { printf 'RUN_ROOT 不存在: %s\n' "$RUN_ROOT" >&2; exit 2; }
origin="$PWD"
out=$(mktemp -d "${TMPDIR:-/tmp}/c5-evidence.XXXXXXXX") || exit 2
mkdir -p "$out/meta" "$out/device-export"
list="$out/meta/sources.nul"
status="$out/meta/collection-status.txt"
touch "$list" "$status"
add() {
  if [[ -e "$1" && -r "$1" ]]; then printf '%s\0' "$1" >> "$list";
  else printf 'MISSING_OR_UNREADABLE %s\n' "$1" >> "$status"; fi
}
snap() {
  local label="$1" rc; shift
  timeout 60 "$@" > "$out/meta/$label.txt" 2>&1; rc=$?
  printf 'COMMAND %s exit=%s\n' "$label" "$rc" >> "$status"
}
add "$RUN_ROOT"
printf 'collected_utc=%s\nfault_utc=2026-10-13 08:27:47..08:27:51\nfault_beijing=2026-10-13 16:27:47..16:27:51\nRUN_ROOT=%s\norigin=%s\n' \
  "$(date -u +%FT%TZ)" "$RUN_ROOT" "$origin" > "$out/meta/context.txt"
snap uname uname -a
snap npu-smi npu-smi info
snap ports ss -ltnp
snap processes ps -eo pid,ppid,lstart,comm
snap dmesg dmesg -T
snap journal journalctl --utc --since '2026-10-13 08:27:30' --until '2026-10-13 08:28:10' --no-pager
grep -q -- '-- No entries --' "$out/meta/journal.txt" && printf 'NO_JOURNAL_ENTRIES_IN_FAULT_WINDOW\n' >> "$status"
env | LC_ALL=C sort | grep -E '^(CANN_ROOT|VIRTUAL_ENV|LD_LIBRARY_PATH|PYTHONPATH|ASCEND_(HOME_PATH|TOOLKIT_HOME|PROCESS_LOG_PATH|WORK_PATH)|NPU_COLLECT_PATH|HCCL_[A-Z0-9_]+|AFD_HCCL_[A-Z0-9_]+|EXECUTION_MODE|U_BATCHES|ENABLE_PD|ENABLE_DSPARK)=' > "$out/meta/shell-env.txt" || true
snap packages "${PYTHON_BIN:-python3}" -m pip show torch torch-npu vllm vllm-ascend vllm-afd-plugin
roots=("$origin" "$RUN_ROOT" /root /home/z00569729 /tmp)
[[ -z "${C5_TASK_DIR:-}" ]] || roots+=("$C5_TASK_DIR")
logroots=(/root/ascend/log "${HOME}/ascend/log")
for key in NPU_COLLECT_PATH ASCEND_WORK_PATH; do
  [[ -z "${!key:-}" ]] || roots+=("${!key}")
done
[[ -z "${ASCEND_PROCESS_LOG_PATH:-}" ]] || logroots+=("$ASCEND_PROCESS_LOG_PATH")
[[ -z "${ASCEND_WORK_PATH:-}" ]] || logroots+=("$ASCEND_WORK_PATH/plog" "$ASCEND_WORK_PATH/atrace")
pids=(83688 83687 3155455 3155454)
while read -r pid; do pids+=("$pid"); done < <(grep -hoE 'pid=[0-9]+' "$RUN_ROOT"/*.log 2>/dev/null | cut -d= -f2 | sort -nu)
for pid in "${pids[@]}"; do
  if [[ ! -r "/proc/$pid/maps" ]]; then printf 'PROCESS_GONE_OR_UNREADABLE %s\n' "$pid" >> "$status"; continue; fi
  if ! grep -qE 'torch_npu|/vllm|afd-plugin' "/proc/$pid/maps"; then printf 'PID_REUSED_OR_NOT_NPU_WORKER %s\n' "$pid" >> "$status"; continue; fi
  snap "pid-$pid-maps" cat "/proc/$pid/maps"
  readlink -f "/proc/$pid/cwd" > "$out/meta/pid-$pid-cwd.txt" 2>&1 || true
  tr '\0' '\n' < "/proc/$pid/cmdline" > "$out/meta/pid-$pid-cmdline.txt" 2>/dev/null || true
  tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^(CANN_ROOT|VIRTUAL_ENV|LD_LIBRARY_PATH|PYTHONPATH|ASCEND_(HOME_PATH|TOOLKIT_HOME|PROCESS_LOG_PATH|WORK_PATH)|NPU_COLLECT_PATH|HCCL_[A-Z0-9_]+|AFD_HCCL_[A-Z0-9_]+|EXECUTION_MODE|U_BATCHES|ENABLE_PD|ENABLE_DSPARK)=' > "$out/meta/pid-$pid-env.txt" || true
  cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null); [[ -z "$cwd" ]] || roots+=("$cwd")
  while IFS='=' read -r key value; do
    case "$key" in
      ASCEND_PROCESS_LOG_PATH) logroots+=("$value");;
      ASCEND_WORK_PATH) roots+=("$value"); logroots+=("$value/plog" "$value/atrace");;
      NPU_COLLECT_PATH) roots+=("$value");;
    esac
  done < "$out/meta/pid-$pid-env.txt"
done
for key in AFD_PLUGIN_ROOT VLLM_ROOT VLLM_ASCEND_ROOT; do
  repo="${!key:-}"; [[ -d "$repo" ]] || continue
  roots+=("$repo")
  snap "$key-head" git -C "$repo" rev-parse HEAD
  snap "$key-status" git -C "$repo" status --short --branch
done
tool=$(command -v msnpureport 2>/dev/null || true)
[[ -n "$tool" ]] || tool=/usr/local/Ascend/driver/tools/msnpureport
if [[ -x "$tool" ]]; then
  snap msnpureport-help "$tool" --help
  report_args=(-f)
  grep -Eq '^[[:space:]]*report([[:space:]]|$)' "$out/meta/msnpureport-help.txt" && report_args=(report -f)
  printf 'Exporting device logs; timeout 180 seconds...\n'
  (cd "$out/device-export" && timeout 180 "$tool" "${report_args[@]}") > "$out/meta/msnpureport.txt" 2>&1
  report_rc=$?
  printf 'COMMAND msnpureport exit=%s\n' "$report_rc" >> "$status"
  if ((report_rc != 0)) && grep -Fq 'msnpureport in docker should be used with --docker' "$out/meta/msnpureport.txt"; then
    printf 'Retrying device log export with --docker; timeout 180 seconds...\n'
    (cd "$out/device-export" && timeout 180 "$tool" "${report_args[@]}" --docker) > "$out/meta/msnpureport-docker.txt" 2>&1
    report_rc=$?
    printf 'COMMAND msnpureport-docker exit=%s\n' "$report_rc" >> "$status"
  fi
  printf 'DEVICE_EXPORT_FINAL_EXIT=%s\n' "$report_rc" >> "$status"
  ((report_rc == 0)) || printf 'DEVICE_EXPORT_FAILED; see meta/msnpureport*.txt; host collection may be needed\n' >> "$status"
  timeout 60 find "$out/device-export" -type f -print0 > "$out/meta/device-export-files.nul" 2>> "$status"
  printf 'DEVICE_EXPORT_FILE_SEARCH exit=%s\n' "$?" >> "$status"
  [[ -s "$out/meta/device-export-files.nul" ]] || printf 'NO_DEVICE_EXPORT_FILES\n' >> "$status"
else printf 'MISSING_TOOL msnpureport; host collection may be needed\n' >> "$status"; fi
for path in /var/log/npu/report /var/log/npu/slog /var/log/ascend_seclog /usr/local/Ascend/driver/version.info /usr/local/Ascend/driver/ascend_install.info /etc/ascend_install.info; do add "$path"; done
cann="${CANN_ROOT:-/usr/local/Ascend/cann-9.2.0}"
[[ ! -d "$cann" ]] || timeout 60 find -L "$cann" -maxdepth 4 -type f \( -name '*install.info' -o -name 'version.info' \) -print0 >> "$list" 2>> "$status"
for path in "${logroots[@]}"; do
  [[ -d "$path" ]] || { printf 'MISSING_LOG_ROOT %s\n' "$path" >> "$status"; continue; }
  for pid in "${pids[@]}"; do
    timeout 60 find "$path" -maxdepth 5 -type f \( -name "plog-${pid}_*" -o -name "device-${pid}_*" \) -print0 >> "$list" 2>> "$status"
  done
done
printf 'Searching existing extra-info/atrace; no model files are collected...\n'
for path in "${roots[@]}"; do
  [[ -d "$path" ]] || continue
  timeout 60 find "$path" -maxdepth 7 \( -name .git -o -name .cache -o -name models -o -name model_weights -o -path "$out" \) -prune -o \
    -type d \( -name extra-info -o -name atrace \) -print0 -prune >> "$out/meta/dump-paths.nul" 2>> "$status"
  printf 'SEARCH %s exit=%s\n' "$path" "$?" >> "$status"
done
[[ ! -f "$out/meta/dump-paths.nul" ]] || cat "$out/meta/dump-paths.nul" >> "$list"
[[ ! -s "$out/meta/dump-paths.nul" ]] && printf 'MISSING_EXCEPTION_DUMP_AND_ATRACE_IN_SEARCHED_ROOTS\n' >> "$status"
touch "$out/meta/extra-info-files.nul"
while IFS= read -r -d '' path; do
  [[ "$path" == */extra-info ]] || continue
  timeout 60 find "$path" -type f -print0 >> "$out/meta/extra-info-files.nul" 2>> "$status"
  printf 'EXTRA_INFO_FILE_SEARCH %s exit=%s\n' "$path" "$?" >> "$status"
done < "$out/meta/dump-paths.nul"
sort -zu "$out/meta/extra-info-files.nul" -o "$out/meta/extra-info-files.nul"
tr '\0' '\n' < "$out/meta/extra-info-files.nul" > "$out/meta/extra-info-files.txt"
[[ -s "$out/meta/extra-info-files.nul" ]] || printf 'NO_EXTRA_INFO_FILES_IN_SEARCHED_ROOTS; directories or atrace alone do not establish a saved exception dump\n' >> "$status"
sort -zu "$list" -o "$list"
tr '\0' '\n' < "$list" > "$out/meta/source-inventory.txt"
archive="${out}.tar.gz"
tar --null -czf "$archive" -T "$list" -C "$(dirname "$out")" "$(basename "$out")" 2> "${out}.tar-status.txt"
rc=$?
if ((rc > 1)); then printf 'Archive failed: %s\nSee %s.tar-status.txt\n' "$archive" "$out" >&2; exit "$rc"; fi
sha256sum "$archive" > "${archive}.sha256"
printf '\nOUTPUT=%s\nSHA256=%s.sha256\nSTATUS=%s\nTAR_STATUS=%s.tar-status.txt\n' "$archive" "$archive" "$status" "$out"
cat "$status"
