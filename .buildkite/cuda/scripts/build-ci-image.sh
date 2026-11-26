#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

# Build and push the AFD Plugin CI Docker image with registry-based layer caching.
# One image serves every CUDA step: it carries the plugin's dev/E2E groups and
# the `afdep` extra (afdEP-comm, the DeepEPM2NAFDConnector transport, pinned by
# git commit in uv.lock), on top of vLLM and the DeepGEMM bundled in vLLM.
#
# Usage: build-ci-image.sh
#
# Expects the following environment variables (set by Buildkite):
#   BUILDKITE_COMMIT          - commit SHA used as the image tag
ECR_NAMESPACE="public.ecr.aws/q9t5s3a7"
REGISTRY="${ECR_NAMESPACE}/vllm-ci-test-repo"
REGION="us-east-1"
DOCKERFILE="docker/Dockerfile.ci"
BUILDER_NAME="afd-plugin-builder"

# Ensure that the env vars are actually set, otherwise exit early
if [ -z "${BUILDKITE_COMMIT:-}" ]; then
    echo "ERROR: BUILDKITE_COMMIT is not set"
    exit 1
fi
echo "BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"

# Authenticate to ECR Public
aws ecr-public get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_NAMESPACE"

# Compute the cache tag for the dependencies image as the hash of the contents.
# Dockerfile is included so base-image / vLLM tag changes invalidate the cache;
# uv.lock covers the afdEP-comm commit, so a new pin rebuilds the deps layer.
DEP_FILES=(pyproject.toml setup.py uv.lock docker/Dockerfile.ci)
CACHE_KEY=$(cat "${DEP_FILES[@]}" | sha256sum | cut -c1-16)
CACHE_TAG="deps-cache-${CACHE_KEY}"
echo "Cache key: ${CACHE_TAG}"

# Set up buildx with docker-container driver; we need to do this
# since cache export is not supported for the default docker driver
# that is running in the CI.
docker buildx inspect $BUILDER_NAME >/dev/null 2>&1 \
|| docker buildx create --name $BUILDER_NAME --driver docker-container
docker buildx use $BUILDER_NAME

echo "Building image tag: ${REGISTRY}:${BUILDKITE_COMMIT}"

docker buildx build --push --progress=plain \
    --cache-from "type=registry,ref=${REGISTRY}:${CACHE_TAG}" \
    --cache-to "type=registry,ref=${REGISTRY}:${CACHE_TAG},mode=max,compression=zstd" \
    --file "$DOCKERFILE" \
    -t "${REGISTRY}:${BUILDKITE_COMMIT}" .
