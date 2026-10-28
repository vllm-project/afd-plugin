#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Drive load against a running AFD instance (e.g. the attention endpoint of a
# recipe under recipe/gpu/). Thin wrapper around `vllm bench serve`; every knob
# is overridable via an environment variable. For a decode-only run over a
# warmed prompt pool, use decode_bench_client.py instead.
#
# Usage:
#   MODEL_PATH=/path/to/weights tools/benchmarks/request_generator.sh
#   HOST=127.0.0.1 PORT=18305 INPUT_LEN=8192 OUTPUT_LEN=256 MAX_CONCURRENCY=64 \
#       MODEL_PATH=/path/to/weights tools/benchmarks/request_generator.sh
#
# Variables (default):
#   MODEL_PATH        model weights dir/name              (required)
#   HOST              server host                         (127.0.0.1)
#   PORT              server port                         (18305)
#   DATASET_NAME      vllm bench dataset                  (random)
#   INPUT_LEN         random input length (ISL)           (1024)
#   OUTPUT_LEN        random output length (OSL)          (128)
#   NUM_PROMPTS       total prompts to send               (1024)
#   REQUEST_RATE      requests/sec (inf = as fast as can) (5)
#   MAX_CONCURRENCY   max in-flight requests              (32)
#   RESULT_DIR        directory for saved results         (/tmp/results)
#   RESULT_FILENAME   result json filename                (decode_bench.json)
#   EXTRA_ARGS        extra args appended to the command  (empty)
set -euo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the model weights dir/name}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18305}
DATASET_NAME=${DATASET_NAME:-random}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-1024}
REQUEST_RATE=${REQUEST_RATE:-5}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-32}
RESULT_DIR=${RESULT_DIR:-/tmp/results}
RESULT_FILENAME=${RESULT_FILENAME:-decode_bench.json}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$RESULT_DIR"

# shellcheck disable=SC2086
uv run vllm bench serve \
    --host "$HOST" --port "$PORT" \
    --model "$MODEL_PATH" \
    --dataset-name "$DATASET_NAME" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --result-dir "$RESULT_DIR" \
    --result-filename "$RESULT_FILENAME" \
    --save-result \
    $EXTRA_ARGS
