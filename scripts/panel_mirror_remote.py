"""Remote side of rlforge panel mirror. Runs on the training node (360-2).

Usage: python3 panel_mirror_remote.py <run_id> <step_offset>
Prints one JSON bundle to stdout:
  metrics  -> report_watch --once extraction (steps / task_bins)
  run      -> run.json metadata
  evals    -> [{step_global, overall, by_task, path}] summary-only per eval300 json
  running  -> whether a trainer process for this run is alive
  rollout_samples_path -> path to rollout_samples.jsonl (or null)
"""
import json, pathlib, re, subprocess, sys

ROOT = pathlib.Path('/data/home/guoshaoyang/aiq_rl')
VPY = str(ROOT / 'venv/bin/python')
EVAL_DIRS = [ROOT / 'evals', pathlib.Path('/data/shared/guoshaoyang/aiq_rl_store/evals')]


def main() -> None:
    run_id, offset = sys.argv[1], int(sys.argv[2])
    tag_m = re.search(r'step(\d+)', run_id)
    tag = 'step' + tag_m.group(1) if tag_m else run_id

    # 1) metrics extraction
    out_dir = pathlib.Path('/tmp/panel_export') / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer_log = ROOT / ('logs/trainer_dp_' + run_id[len('async_dp_'):] + '.log')
    metrics = None
    if trainer_log.is_file():
        subprocess.run([VPY, str(ROOT / 'report_watch.py'), '--run', str(ROOT / 'runs' / run_id),
                        '--trainer-log', str(trainer_log), '--out', str(out_dir), '--once'],
                       capture_output=True, text=True)
        mj = out_dir / 'metrics.json'
        if mj.is_file():
            metrics = json.load(open(mj))

    # 2) run metadata
    run_json = None
    rj = ROOT / 'runs' / run_id / 'run.json'
    if rj.is_file():
        run_json = json.load(open(rj))

    # 3) eval summaries (summary-only; full records pulled on demand by local side)
    evals, seen = [], set()
    for ed in EVAL_DIRS:
        if not ed.is_dir():
            continue
        for f in sorted(ed.glob(f'*{tag}*ckpt*_eval300_*.json')):
            gm = re.search(r'_global(\d+)_', f.name)
            cm = re.search(r'ckpt(\d+)', f.name)
            if gm:
                step_global = int(gm.group(1))
            elif cm:
                step_global = int(cm.group(1)) + offset
            else:
                continue
            if step_global in seen:
                continue
            seen.add(step_global)
            try:
                d = json.load(open(f))
                evals.append({'step_global': step_global, 'overall': d.get('overall', {}),
                              'by_task': d.get('by_task', {}), 'path': str(f)})
            except Exception:
                continue

    # 4) liveness + rollout samples path
    pg = subprocess.run(['pgrep', '-f', f'trainer.*{run_id}'], capture_output=True, text=True)
    running = bool(pg.stdout.strip())
    rsp = ROOT / 'runs' / run_id / 'rollout_samples.jsonl'

    json.dump({'run_id': run_id, 'metrics': metrics, 'run': run_json,
               'evals': sorted(evals, key=lambda e: e['step_global']),
               'running': running,
               'rollout_samples_path': str(rsp) if rsp.is_file() else None},
              sys.stdout)


if __name__ == '__main__':
    main()
