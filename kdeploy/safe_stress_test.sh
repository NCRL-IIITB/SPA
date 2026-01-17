#!/bin/bash
set -euo pipefail

# Safe CPU, memory, and disk stress test loop
echo "Starting ultra-safe stress test (CPU + memory + controlled disk)..."

# Detect available CPU and memory resources
CPU_CORES=$(nproc)
MEM_TOTAL_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_TOTAL_MB=$((MEM_TOTAL_KB / 1024))
MEM_USE_MB=$((MEM_TOTAL_MB * 80 / 100))

# Disk usage limit per cycle (10 GB)
DISK_LIMIT_MB=$((10 * 1024))

echo "Detected:"
echo "Total Memory: ${MEM_TOTAL_MB} MB → Using ${MEM_USE_MB} MB"
echo "CPU Cores: ${CPU_CORES}"
echo "Disk Limit: ${DISK_LIMIT_MB} MB per cycle"

# Temporary directory for stress-ng disk writes
TMP_DIR="/tmp/stress_safe_tmp"
mkdir -p "$TMP_DIR"

cleanup() {
  echo "Cleaning up temporary files..."
  rm -rf "$TMP_DIR"/* || true
}
trap cleanup EXIT

# Run repeated stress cycles with cleanup between runs
CYCLE=1
while true; do
  echo "Starting stress cycle $CYCLE at $(date)"
  cleanup

  # Run CPU + memory + disk test (max 10 GB temp files)
  stress-ng \
    --cpu "$CPU_CORES" \
    --vm 4 \
    --vm-bytes "${MEM_USE_MB}M" \
    --hdd 1 \
    --hdd-bytes "${DISK_LIMIT_MB}M" \
    --temp-path "$TMP_DIR" \
    --timeout 30m \
    --metrics-brief

  cleanup
  echo "Cycle $CYCLE completed safely at $(date)"
  echo "Sleeping 60 s before next cycle..."
  sleep 60
  ((CYCLE++))
done

