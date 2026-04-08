#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   HF_TOKEN=hf_xxx ./publish_to_huggingface.sh <hf_username_or_org>

if [[ $# -lt 1 ]]; then
  echo "usage: HF_TOKEN=hf_xxx $0 <hf_username_or_org>"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required"
  exit 1
fi

OWNER="$1"
REPO_ID="${OWNER}/das-smollm2-360m-das-v1"

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
hf repo create "$REPO_ID" --repo-type model --private=false || true
hf upload "$REPO_ID" . --repo-type model --commit-message "Release DAS distilled SmolLM2-360M v1"

echo "Published to https://huggingface.co/${REPO_ID}"
