#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

config_path="config/manual_mode/xarm7_manual_record_config.yaml"
dataset_root="$(
  sed -n '/^dataset:/,/^[^[:space:]]/s/^[[:space:]]*root:[[:space:]]*//p' "$config_path" \
    | head -n 1 \
    | tr -d "\"'"
)"

if [[ -z "$dataset_root" ]]; then
  printf 'Could not read dataset.root from %s\n' "$config_path" >&2
  exit 1
fi

if [[ "$dataset_root" != /* ]]; then
  dataset_root="$repo_root/$dataset_root"
fi

record_args=("$@")
resume_requested=false
for arg in "${record_args[@]}"; do
  if [[ "$arg" == "-r" ]]; then
    resume_requested=true
    break
  fi
done

if [[ -e "$dataset_root" ]]; then
  if [[ ! -f "$dataset_root/meta/info.json" ]]; then
    printf 'Dataset directory exists but is not a valid LeRobot dataset: %s\n' "$dataset_root" >&2
    printf 'Choose a new dataset.root, or remove this empty/incomplete directory before recording.\n' >&2
    exit 1
  fi

  if [[ "$resume_requested" == false ]]; then
    record_args=("-r" "${record_args[@]}")
  fi
elif [[ "$resume_requested" == true ]]; then
  printf 'Cannot resume because the dataset directory does not exist: %s\n' "$dataset_root" >&2
  exit 1
fi

exec uv run uf-lerobot-record \
  --config_path "$config_path" \
  "${record_args[@]}"
