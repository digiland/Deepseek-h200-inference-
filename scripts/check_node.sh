#!/usr/bin/env bash
# Collect everything about the H200 node that affects inference performance.
# Run once before benchmarking and send me the output file:
#   scripts/check_node.sh > results/node_info.txt 2>&1
set -uo pipefail

section() { printf '\n===== %s =====\n' "$*"; }

section "GPUs"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit,clocks.max.sm,clocks.max.mem,persistence_mode,mig.mode.current,ecc.mode.current --format=csv

section "Driver / CUDA"
nvidia-smi | head -4
command -v nvcc >/dev/null && nvcc --version | tail -2

section "Topology (want NV18 between every GPU pair)"
nvidia-smi topo -m

section "NVLink status GPU0"
nvidia-smi nvlink --status -i 0 2>/dev/null | head -20

section "Fabric manager (required for NVSwitch systems)"
systemctl is-active nvidia-fabricmanager 2>/dev/null || echo "systemctl unavailable / not running"

section "CPU / NUMA / memory"
lscpu | grep -E 'Model name|Socket|NUMA node|Thread|Core'
free -g
command -v numactl >/dev/null && numactl -H | head -20

section "Clocks currently applied (throttling?)"
nvidia-smi -q -d PERFORMANCE | grep -E 'GPU [0-9]|Idle|Thermal|Power|HW Slowdown' | head -40

section "Python env"
for py in ${PYTHONS:-python python3}; do
  command -v "$py" >/dev/null || continue
  "$py" - <<'EOF'
import importlib, sys
print("python", sys.version.split()[0], sys.executable)
for m in ["torch", "triton", "vllm", "sglang", "flashinfer", "deep_gemm", "deep_ep", "flash_mla", "sgl_kernel", "transformers", "nixl"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:14s} MISSING ({type(e).__name__})")
try:
    import torch
    print("  cuda", torch.version.cuda, "devices", torch.cuda.device_count(),
          [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])
except Exception as e:  # noqa: BLE001
    print("  torch cuda check failed:", e)
EOF
done

section "Disk (model is ~510 GB)"
df -h "${MODEL_PATH:-$HOME}" 2>/dev/null
