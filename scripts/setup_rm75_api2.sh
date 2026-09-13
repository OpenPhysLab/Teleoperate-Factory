#!/usr/bin/env bash
set -euo pipefail
RM75_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${RM75_ROOT}/.rm75_python"
python3 -m pip install --target "${RM75_ROOT}/.rm75_python" \
  -r "${RM75_ROOT}/requirements_rm75_api2.txt"
echo "Dependencies installed. Activate with:"
echo "  source ${RM75_ROOT}/scripts/activate_rm75_api2.sh"
