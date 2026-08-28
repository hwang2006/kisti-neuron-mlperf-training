#!/bin/bash

echo "===== HOST ====="
hostname
hostname -f 2>/dev/null || true

echo
echo "===== OS ====="
grep PRETTY_NAME /etc/os-release 2>/dev/null || true

echo
echo "===== CPU ====="
lscpu | grep -E \
'Model name|Socket\(s\)|Core\(s\) per socket|CPU\(s\)|NUMA node\(s\)' \
|| true

echo
echo "===== MEMORY ====="
free -h

echo
echo "===== GPU ====="
nvidia-smi \
    --query-gpu=index,name,memory.total,driver_version,compute_cap \
    --format=csv,noheader

echo
echo "===== GPU TOPOLOGY ====="
nvidia-smi topo -m

echo
echo "===== SINGULARITY ====="
singularity --version

echo
echo "===== FILESYSTEM ====="
df -h /scratch /home01 2>/dev/null || true
