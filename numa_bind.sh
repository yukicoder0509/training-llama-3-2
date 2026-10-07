#!/bin/bash
# Print a `numactl` prefix that binds to the NUMA node of the job's first GPU, or nothing.
#   NUMA_BIND=local : CPUs = this job's allowed CPUs on the GPUs' node only, memory = that node
#   NUMA_BIND=mem   : keep all allowed CPUs, memory = the GPUs' node only
#   unset / 0       : no binding (Slurm gives 24 CPUs split 12/12 across both sockets; the GPUs sit on one)
# Slurm here ignores --gres-flags=enforce-binding / --sockets-per-node (jobs 505605-505607), hence binding in-job.
mode=${NUMA_BIND:-0}
[ "$mode" = 0 ] && exit 0
bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i 0 | tr 'A-Z' 'a-z' | sed 's/^00000000:/0000:/')
node=$(cat /sys/bus/pci/devices/$bus/numa_node)
if [ "$mode" = mem ]; then
    echo "numactl --membind=$node"
    exit 0
fi
cpus=$(python3 - "$node" <<'PY'
import os, sys
def parse(s):
    out = set()
    for part in s.strip().split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out
node_cpus = parse(open(f"/sys/devices/system/node/node{sys.argv[1]}/cpulist").read())
print(",".join(map(str, sorted(node_cpus & os.sched_getaffinity(0)))))
PY
)
echo "numactl --physcpubind=$cpus --membind=$node"
