#!/usr/bin/env python3
"""Dry-drill make_plan.py: the real ../../make_plan.py output, rewritten for a no-GPU drill under $DRY_ROOT.
Run dirs move to $DRY_ROOT/runs, backends become the mock vLLM servers in $DRY_BACKENDS, the router listens on
$DRY_ROUTER_PORT, and the trainer/worker argv run fake_trainer.py with the real trainer arguments.
Usage: make_plan.py RUN_NAME > plan.json   (same environment knobs as the real one)
"""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NFS_RUNS = '/home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017/runs/'
out = subprocess.run([sys.executable, str(HERE.parents[1] / 'make_plan.py'), sys.argv[1]], check=True,
                     capture_output=True, text=True).stdout
plan = json.loads(out.replace(NFS_RUNS, os.environ['DRY_ROOT'] + '/runs/'))
plan['backends'] = [{'url': u} for u in os.environ['DRY_BACKENDS'].split(',')]
plan.update(min_healthy=len(plan['backends']), backend_wait_s=120, router_port=int(os.environ['DRY_ROUTER_PORT']),
            stall_s=600)


def fake(argv, role):
    return [plan['python'], str(HERE.parent / 'fake_trainer.py'), role] + argv[argv.index('rlforge.trainer') + 1:]


plan['trainer_argv'] = fake(plan['trainer_argv'], 'rank0')
plan['worker_argvs'] = {ip: fake(a, 'worker') for ip, a in plan['worker_argvs'].items()}
print(json.dumps(plan, indent=2))
