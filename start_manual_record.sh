#!/usr/bin/env bash

set -euo pipefail


exec uv run uf-lerobot-record \
  --config_path config/manual_mode/xarm7_manual_record_config.yaml \
  "$@"


