#!/bin/bash
# Kill-line cross-eval driver (frozen weights x killy 170/200).
# Run on the node that holds the run dirs (node_c). Cheap: physics-bound,
# ~1 CPU core + a sliver of one GPU per job, so it shares the box with the
# running arms at low priority (nice).
#
# Usage: bash run_xeval_killy.sh            # all six jobs, parallel
#        SEEDS=0:64 GPUS="6 7" bash run_xeval_killy.sh
set -u
PROJ=${PROJ:-/path/to/suika-dqn}
PY=$PROJ/suika-venv/bin/python3
CODE=$PROJ/suika_dqn
OUT=${OUT:-$PROJ/runs/xeval_killy_$(date +%m%d_%H%M)}
GPUS=${GPUS:-"6 7"}
SEEDS=${SEEDS:-0:32}
NICE=${NICE:-10}

W3C=$PROJ/runs/wave3c_20260929/w3c_tf_xl
CONT=$PROJ/runs/wave4_20260929/w4_k200_cont
mkdir -p "$OUT/src" "$OUT/stage"

# Freeze every weight source first: policy.pt is republished every ~2s, and
# step ckpts roll off after ckpt_keep=4.
cp -f "$W3C/checkpoints/step3801_env482771285.pt" "$OUT/src/fork_step3801.pt" || exit 1
cp -f "$W3C/policy.pt" "$OUT/src/w3c_k170_latest.pt" || exit 1
cp -f "$CONT/policy.pt" "$OUT/src/w4cont_k200_latest.pt" || exit 1
md5sum "$OUT/src/"*.pt > "$OUT/src/MD5SUMS" 2>/dev/null
echo "[xeval] out=$OUT seeds=$SEEDS gpus=$GPUS nice=$NICE"
cat "$OUT/src/MD5SUMS"

# label | killy | ckpt | config
if [ -n "${JOBS_FILE:-}" ]; then
  mapfile -t JOBS < <(grep -v '^[[:space:]]*\(#\|$\)' "$JOBS_FILE")
else
  JOBS=(
    "fork@170|170|$OUT/src/fork_step3801.pt|$CODE/configs/w3c_tf_xl.yaml"
    "fork@200|200|$OUT/src/fork_step3801.pt|$CODE/configs/w3c_tf_xl.yaml"
    "w3c_k170@170|170|$OUT/src/w3c_k170_latest.pt|$CODE/configs/w3c_tf_xl.yaml"
    "w3c_k170@200|200|$OUT/src/w3c_k170_latest.pt|$CODE/configs/w3c_tf_xl.yaml"
    "w4cont_k200@200|200|$OUT/src/w4cont_k200_latest.pt|$CODE/configs/w4_k200_cont.yaml"
    "w4cont_k200@170|170|$OUT/src/w4cont_k200_latest.pt|$CODE/configs/w4_k200_cont.yaml"
  )
fi

i=0
pids=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r label killy ckpt cfg <<< "$job"
  gpu=$(echo $GPUS | awk -v i=$i '{print $((i % NF) + 1)}')
  mkdir -p "$OUT/stage/$label"
  CUDA_VISIBLE_DEVICES=$gpu nice -n $NICE $PY "$CODE/xeval_killy.py" \
      --config "$cfg" --ckpt "$ckpt" --killy "$killy" --seeds "$SEEDS" \
      --label "$label" --stage-dir "$OUT/stage/$label" \
      --out "$OUT/$label.json" \
      > "$OUT/$label.log" 2>&1 &
  pids+=($!)
  echo "[xeval] launched $label on gpu$gpu pid $!"
  i=$((i + 1))
done
for p in "${pids[@]}"; do wait "$p"; done

echo "[xeval] done; summary"
$PY - "$OUT" <<'EOF'
import glob, json, os, sys
out = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(out, "*.json"))):
    try:
        rows.append(json.load(open(p)))
    except Exception as e:
        print(f"  (unreadable {os.path.basename(p)}: {e})")
hdr = (f"{'label':18s} {'killy':>5s} {'wk':>6s} {'digest':>16s} "
       f"{'mean':>7s} {'med':>7s} {'p25':>7s} {'max':>6s} {'p2000':>5s} "
       f"{'fruit':>5s} {'moves':>6s} {'n':>3s}")
print(hdr)
for r in rows:
    print(f"{r['label']:18s} {r['killy']:5d} {r['weights']:>6s} "
          f"{r['obs_digest']:>16s} {r['mean']:7.1f} {r['median']:7.1f} "
          f"{r['p25']:7.1f} {r['max']:6.0f} {r['p2000']:5.2f} "
          f"{r['maxfruit_max']:5d} {r['moves_mean']:6.0f} {r['n']:3d}")
digs = {r["obs_digest"] for r in rows}
print(f"[xeval] distinct obs digests: {len(digs)} "
      f"({'OBS IDENTICAL across kill lines' if len(digs) == 1 else 'MISMATCH!'})")
EOF
