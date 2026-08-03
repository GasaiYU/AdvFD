#!/usr/bin/env bash
# Diagnose the interconnect before optimizing anything else.
#
# Part 1 inspects the node for IB hardware (local, no distributed setup).
# Part 2 measures achieved all-reduce bandwidth across nodes.
#
# Part 1 only -- run on any single node, no rendezvous needed:
#   LOCAL_ONLY=1 bash my_tools/check_nccl_transport.sh
#
# Full check -- run on every node with the same rendezvous as training:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash my_tools/check_nccl_transport.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash my_tools/check_nccl_transport.sh
#
# Optional:
#   GPUS_PER_NODE=8
#   NCCL_DEBUG=INFO          also print NCCL's own transport selection
#   NCCL_IB_HCA=mlx5_0       force a specific HCA
#   NCCL_SOCKET_IFNAME=ib0   force a specific interface
#   OUTPUT_JSON=path.json

set -euo pipefail

echo "=============================================================="
echo " Part 1: does this node have InfiniBand hardware?"
echo "=============================================================="
echo "-- hostname: $(hostname)"

echo
echo "-- ibstat (IB port state; 'Active' means usable):"
if command -v ibstat >/dev/null 2>&1; then
    ibstat 2>&1 | grep -E "CA '|State:|Physical state:|Rate:" || echo "   ibstat produced no port info"
else
    echo "   ibstat not installed"
fi

echo
echo "-- ibv_devinfo (verbs devices NCCL can use):"
if command -v ibv_devinfo >/dev/null 2>&1; then
    ibv_devinfo 2>/dev/null | grep -E "hca_id|state:|link_layer" || echo "   no verbs devices reported"
else
    echo "   ibv_devinfo not installed"
fi

echo
echo "-- IB device nodes under /dev/infiniband:"
ls -1 /dev/infiniband/ 2>/dev/null || echo "   /dev/infiniband missing (no IB, or not mapped into this container)"

echo
echo "-- Mellanox/IB PCI devices:"
if command -v lspci >/dev/null 2>&1; then
    lspci 2>/dev/null | grep -iE "mellanox|infiniband" || echo "   none found via lspci"
else
    echo "   lspci not installed"
fi

echo
echo "-- network interfaces (ib*/roce* suggest IB or RoCE):"
if command -v ip >/dev/null 2>&1; then
    ip -o -4 addr show 2>/dev/null | awk '{printf "   %-12s %s\n", $2, $4}'
else
    echo "   ip not installed"
fi

echo
echo "-- interface speeds (10000 Mb/s = 10 GbE = ~1 GB/s ceiling):"
if [ -d /sys/class/net ]; then
    for dev in /sys/class/net/*; do
        name=$(basename "$dev")
        [ "$name" = "lo" ] && continue
        speed=$(cat "$dev/speed" 2>/dev/null || echo "n/a")
        echo "   ${name}: ${speed} Mb/s"
    done
else
    echo "   /sys/class/net missing (not Linux)"
fi

if [ "${LOCAL_ONLY:-0}" = "1" ]; then
    echo
    echo "LOCAL_ONLY=1, skipping the bandwidth measurement."
    echo "Interpretation:"
    echo "  * ibstat shows an Active port + /dev/infiniband exists -> IB is available."
    echo "    If training still logs NET/Socket, NCCL picked the wrong interface."
    echo "  * No IB anywhere -> Ethernet is all you have; the interface speed above"
    echo "    is your ceiling (10000 Mb/s = 10 GbE = ~1 GB/s)."
    exit 0
fi

: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29531}"
: "${GPUS_PER_NODE:=8}"

if (( NNODES > 1 )) && [ "$MASTER_ADDR" = "127.0.0.1" ]; then
    echo "[ERR] NNODES=$NNODES but MASTER_ADDR=127.0.0.1; use node 0's routable address" >&2
    exit 1
fi

OUTPUT_ARGS=()
if [ -n "${OUTPUT_JSON:-}" ]; then
    OUTPUT_ARGS=(--output_json "$OUTPUT_JSON")
fi

echo
echo "=============================================================="
echo " Part 2: measured all-reduce bandwidth ($NNODES x $GPUS_PER_NODE GPUs)"
echo "=============================================================="

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    my_tools/check_nccl_transport.py "${OUTPUT_ARGS[@]}" "$@"
