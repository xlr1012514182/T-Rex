#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# All data/model options are forwarded to the Python CLI (use --help).
if [[ -n "${LEROBOT_SRC:-}" ]]; then
  export PYTHONPATH="${LEROBOT_SRC}:${PYTHONPATH}"
fi
exec "${PYTHON:-python}" "${PROJECT_ROOT}/utils/convert_inlab_to_lerobot.py" "$@"
