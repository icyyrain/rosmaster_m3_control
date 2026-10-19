#!/usr/bin/env bash
# Run a learner script from this folder in the simulation project's WSL
# environment (official HIL-SERL, JAX with the GPU). From Windows:
#   wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/c/Projects/rosmaster_m3_control_red_rl && \
#       pc_client/red_rl/wsl/run_wsl.sh bc.py --demos datasets/red_sweet_rl_demos/m2_expert.npz"
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sim="${ISAAC_HIL_SERL:-/mnt/c/Projects/isaac-hil-serl}"
python="${HIL_SERL_PYTHON:-/root/.venvs/hil-serl-c32939b/bin/python}"
script="$1"; shift
export PYTHONPATH="$here:$sim/third_party/hil-serl/serl_launcher:$sim${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$here/$script" "$@"
