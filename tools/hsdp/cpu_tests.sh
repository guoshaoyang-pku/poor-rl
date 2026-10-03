#!/usr/bin/env bash
# CPU-only (gloo) correctness tests for bench_hsdp.py. No GPU is touched (CUDA_VISIBLE_DEVICES="").
set -u
source /home/tione/guoshaoyang/a100_rl/env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
D=${HSDP_DIR:-/home/tione/guoshaoyang/a100_rl/wave2/hsdp/review}
B=$D/code/bench_hsdp.py
O=$D/cpu; mkdir -p $O; cd $O; rm -f *.pt *.jsonl *.err
COMMON="--device cpu --tiny 64 --layers 4 --fixed-seq 40,56,48,64 --micro 2 --warmup 1 --steps 1 --ar-bench 0"
run() { name=$1; shift; echo "### $name: $*" ; "$@" > $O/$name.log 2>&1; echo "rc=$? $name"; }
run ref   torchrun --nproc_per_node 1 --master_port 29740 $B $COMMON --ref-world 4 --check-grads $O/ref.pt --out $O/ref.jsonl
# torchrun needs WORLD/LOCAL env: single host emulating R=2 x S=2
run hsdp_native torchrun --nproc_per_node 4 --master_port 29741 $B $COMMON --replicate 2 --sync native --check-grads $O/hsdp_native.pt --out $O/hsdp_native.jsonl
run hsdp_every  torchrun --nproc_per_node 4 --master_port 29742 $B $COMMON --replicate 2 --sync every  --check-grads $O/hsdp_every.pt  --out $O/hsdp_every.jsonl
run fsdp_r1     torchrun --nproc_per_node 4 --master_port 29743 $B $COMMON --replicate 1 --sync native --check-grads $O/fsdp_r1.pt --out $O/fsdp_r1.jsonl
run rep4_s1     torchrun --nproc_per_node 4 --master_port 29744 $B $COMMON --replicate 4 --sync native --check-grads $O/rep4_s1.pt --out $O/rep4_s1.jsonl
python - <<'PY'
import torch, json, glob
ref = torch.load("ref.pt")
for n in ["hsdp_native", "hsdp_every", "fsdp_r1", "rep4_s1"]:
    try:
        g = torch.load(f"{n}.pt")
    except Exception as e:
        print(n, "NO GRADS", e); continue
    assert set(g) == set(ref), (n, len(g), len(ref))
    num = max(float((g[k] - ref[k]).abs().max()) for k in ref)
    rel = max(float((g[k] - ref[k]).norm() / (ref[k].norm() + 1e-30)) for k in ref)
    zero = sum(1 for k in ref if float(ref[k].abs().max()) == 0)
    print(f"{n}: params={len(ref)} max_abs_diff={num:.3e} max_rel_l2={rel:.3e} zero_ref_grads={zero}")
for n in ["hsdp_native", "hsdp_every", "fsdp_r1", "rep4_s1"]:
    for l in open(f"{n}.jsonl"):
        r = json.loads(l); print(n, r["status"], "R,S", r["replicate"], r["shard"], "colls/micro-step", r.get("fsdp_colls_rank0"), "gn", [round(s["grad_norm"], 6) for s in r["steps"]])
PY
