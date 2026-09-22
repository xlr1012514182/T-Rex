#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Generic upstream training entry point. For Revo use revo3_v1_trex.py.
# NUM_PROCESSES is the total number of processes across NUM_MACHINES.
NUM_MACHINES="${NUM_MACHINES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_MACHINES * 8))}"
exec "${PYTHON:-python}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG:-${PROJECT_ROOT}/config/sft_qwen.yaml}" \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK:-0}" \
  --main_process_ip "${MASTER_ADDR:-127.0.0.1}" \
  --main_process_port "${MASTER_PORT:-29500}" \
  --deepspeed_multinode_launcher standard \
  "${PROJECT_ROOT}/scripts/train.py" "$@"
