#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/connect_quest_wifi.sh 172.16.204.100
# If USB debugging is currently available, the script enables tcpip mode first.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB_BIN="${ADB_BIN:-${ROOT_DIR}/.tools/platform-tools/adb}"
QUEST_IP="${1:-172.16.204.100}"
QUEST_PORT="${QUEST_PORT:-5555}"

if [ ! -x "${ADB_BIN}" ]; then
  echo "adb not found: ${ADB_BIN}" >&2
  echo "Run the workspace setup or set ADB_BIN=/path/to/adb" >&2
  exit 2
fi

"${ADB_BIN}" start-server >/dev/null
WIRELESS_SERIAL="${QUEST_IP}:${QUEST_PORT}"
WIRELESS_STATE="$("${ADB_BIN}" devices | awk -v target="${WIRELESS_SERIAL}" '$1 == target {print $2; exit}')"
if [ "${WIRELESS_STATE}" = "device" ]; then
  echo "Quest wireless ADB is already connected: ${WIRELESS_SERIAL}"
  "${ADB_BIN}" devices -l
  exit 0
fi

USB_SERIAL="$("${ADB_BIN}" devices | awk 'NR > 1 && $2 == "device" && $1 !~ /:/ {print $1; exit}')"
if [ -n "${USB_SERIAL}" ]; then
  "${ADB_BIN}" -s "${USB_SERIAL}" tcpip "${QUEST_PORT}"
  sleep 1
fi

"${ADB_BIN}" connect "${QUEST_IP}:${QUEST_PORT}"
"${ADB_BIN}" devices -l

for _ in $(seq 1 15); do
  if "${ADB_BIN}" -s "${QUEST_IP}:${QUEST_PORT}" shell true >/dev/null 2>&1; then
    echo "Quest wireless ADB is ready: ${QUEST_IP}:${QUEST_PORT}"
    exit 0
  fi
  echo "Waiting for Quest network-debugging authorization..." >&2
  sleep 1
done
echo "Wireless ADB is not authorized; accept the debugging prompt in Quest." >&2
exit 1
