#!/usr/bin/env bash
# =============================================================================
# IEEE SANER 2027 Replication Package - One-Click Reproduction Runner (Linux/macOS)
# =============================================================================
set -e

echo "============================================================================="
echo "Starting IEEE SANER 2027 Replication Package Verification..."
echo "============================================================================="

python3 scripts/reproduce_all.py

echo ""
echo "[SUCCESS] Full replication suite completed successfully."
