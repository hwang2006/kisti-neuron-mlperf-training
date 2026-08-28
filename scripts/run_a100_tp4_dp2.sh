#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec /bin/bash "$SCRIPT_DIR/run_a100.sh" tp4_dp2
