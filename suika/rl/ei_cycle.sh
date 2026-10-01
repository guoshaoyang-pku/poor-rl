#!/bin/bash
# EI cycle driver (runs on the mac, loops forever):
#   dump top episodes on perm -> transfer shards -> distill on t1 ->
#   64-seed eval on t1 -> if better than current champion, swap anchor on perm
#   and bump the champion. Usage: bash ei_cycle.sh <cycle_start>
set -u
CYCLE=${1:-2}
PERM=node_d
T1=node_c
PDIR=/data/user/suika-dqn
TDIR=/path/to/suika-dqn
RUNDIR=$PDIR/runs/w5rl_20261001/w5_rl_qwen_perm_v4
WORK=/tmp/ei_cycle
mkdir -p $WORK
LOG=$WORK/cycle.log

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

while true; do
  C=ei_c${CYCLE}
  say "=== cycle $CYCLE start ==="
  # 1. dump top episodes from the live RL run (80s on 48 workers)
  ssh $PERM "cd $PDIR && PYTHONPATH=suika_dqn OMP_NUM_THREADS=1 nice -n 10 \
    suika-venv/bin/python suika_dqn/ei_dump.py --run-dir $RUNDIR \
    --out-dir $PDIR/runs/w5rl_20261001/ei_data_c${CYCLE} \
    --top-frac 0.10 --max-episodes 4000 --workers 48" >> $LOG 2>&1 \
    || { say "dump failed, retry in 30m"; sleep 1800; continue; }
  # 2. extract latest RL trainable weights (for distill init) on perm
  ssh $PERM "cd $PDIR && suika-venv/bin/python - <<'EOF' >> $LOG 2>&1
import torch, glob, os
cks = sorted(glob.glob('$RUNDIR/checkpoints/step*.pt'),
             key=os.path.getmtime)
src = cks[-1]
ck = torch.load(src, map_location='cpu', weights_only=False)
sd = ck['state_dict'] if 'state_dict' in ck else ck
tr = {k: v for k, v in sd.items()
      if 'lora_' in k or 'v_head' in k or 'a_head' in k
      or 'pi_head' in k}
torch.save({'state_dict': tr, 'grad_steps': ck.get('grad_steps', 0),
            'trainable_only': True}, '$PDIR/runs/w5rl_20261001/ei_init_latest.pt')
print('init extracted from', src, len(tr), 'keys')
EOF"
  # 3. transfer shards + init to t1 (via local)
  mkdir -p $WORK/data && rm -f $WORK/data/*.npz
  rsync -az $PERM:$PDIR/runs/w5rl_20261001/ei_data_c${CYCLE}/ $WORK/data/ >> $LOG 2>&1 \
    || { say "shard transfer failed"; sleep 1800; continue; }
  ssh $T1 "rm -rf $TDIR/runs/ei_data_c${CYCLE} && mkdir -p $TDIR/runs/ei_data_c${CYCLE}"
  (cd $WORK/data && tar czf - .) | ssh $T1 "tar xzf - -C $TDIR/runs/ei_data_c${CYCLE}" >> $LOG 2>&1
  rsync -az $PERM:$PDIR/runs/w5rl_20261001/ei_init_latest.pt $WORK/ >> $LOG 2>&1
  cat $WORK/ei_init_latest.pt | ssh $T1 "cat > $TDIR/runs/ei_rl_ckpt/ei_init_latest.pt"
  say "data+init on t1"
  # 4. distill on t1 (8 GPUs, ~45 min)
  ssh $T1 "cd $TDIR && rm -rf runs/w6ei_20261001/c${CYCLE} && \
    PYTHONPATH=suika_dqn suika-venv/bin/torchrun --nproc_per_node=8 \
    suika_dqn/bc_learner_qwen.py --config suika_dqn/configs/w6ei_qwen_t1.yaml \
    --run-dir runs/w6ei_20261001/c${CYCLE} --data-dir runs/ei_data_c${CYCLE} \
    --init-from runs/ei_rl_ckpt/ei_init_latest.pt" >> $LOG 2>&1 \
    || { say "distill failed"; sleep 1800; continue; }
  # 5. 64-seed eval of the distilled policy on t1 (single GPU).
  # eval_qwen_policy loops after the first eval -> timeout kill is EXPECTED;
  # success = eval.jsonl has a row.
  ssh $T1 "cd $TDIR && rm -rf runs/w6ei_20261001/ev_c${CYCLE} && \
    PYTHONPATH=suika_dqn CUDA_VISIBLE_DEVICES=0 timeout 2400 \
    suika-venv/bin/python suika_dqn/eval_qwen_policy.py \
    --config suika_dqn/configs/w6ei_qwen_t1.yaml \
    --run-dir runs/w6ei_20261001/ev_c${CYCLE} --obs-dim 800 --seeds 0:64 \
    --ckpt $TDIR/runs/w6ei_20261001/c${CYCLE}/policy.pt --decode qhead" \
    >> $LOG 2>&1
  MEAN=$(ssh $T1 "tail -1 $TDIR/runs/w6ei_20261001/ev_c${CYCLE}/eval.jsonl 2>/dev/null" \
         | python3 -c "import json,sys; s=sys.stdin.read().strip(); print(round(json.loads(s)['mean'],1)) if s else print('')" 2>/dev/null)
  if [ -z "$MEAN" ]; then say "eval produced no result"; sleep 1800; continue; fi
  say "cycle $CYCLE distilled eval mean = $MEAN (64 seeds)"
  # 6. compare with champion; if better, ship trainable ckpt to perm + swap anchor
  CHAMP=$(cat $WORK/champion_mean 2>/dev/null || echo 0)
  BETTER=$(python3 -c "print(1 if $MEAN > $CHAMP else 0)")
  if [ "$BETTER" = "1" ]; then
    CKPT=$(ssh $T1 "ls -t $TDIR/runs/w6ei_20261001/c${CYCLE}/checkpoints/step*.pt | head -1")
    rsync -az $T1:$CKPT $WORK/ei_c${CYCLE}_trainable.pt >> $LOG 2>&1
    cat $WORK/ei_c${CYCLE}_trainable.pt | ssh $PERM "cat > $PDIR/runs/w5rl_20261001/anchor_next.pt && mv $PDIR/runs/w5rl_20261001/anchor_next.pt $PDIR/runs/w5rl_20261001/anchor_teacher.pt"
    echo $MEAN > $WORK/champion_mean
    echo $CYCLE > $WORK/champion_cycle
    say "NEW CHAMPION: cycle $CYCLE mean $MEAN -> anchor swapped on perm"
  else
    say "cycle $CYCLE ($MEAN) did not beat champion ($CHAMP); anchor unchanged"
  fi
  CYCLE=$((CYCLE + 1))
done
