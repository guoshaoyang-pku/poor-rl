#!/usr/bin/env python3
"""Dry-drill stand-in for trainer rank 0, the node-1 worker and the live eval (no GPU, no model).
Behaviour per run name comes from DRY_ROOT/scenario.json:
  rank0  oom (print a CUDA OOM, then hang until the head kills it) | rc1 (exit 1) | done (exit 0) |
         sigterm_head (SIGTERM the head, like an operator stop) | run (keep stepping)
  worker follow (exit 1 once FAILED.json appears, 0 after rank0 is done) | hang (only exits when killed) |
         exit (exit 1 after worker_exit_s, like a rank crash on node 1)
rank0 checkpoints every --save-steps like the trainer (adapter + rollout_state.json) and loads trl-policy-v{step}
on every backend, unloading v{step-4}; a name still loaded from an earlier run is a collision (exit 4).
Usage: fake_trainer.py rank0|worker TRAINER_ARGS...  |  fake_trainer.py eval RUN_DIR
"""
import json, os, signal, sys, time, urllib.error, urllib.request
from pathlib import Path

role = sys.argv[1]
if role == 'eval':
    run = Path(sys.argv[2])
    print('fake eval for', run, flush=True)
    while not (run / 'FAILED.json').exists() and not (run / 'DONE.json').exists():
        time.sleep(2)
    sys.exit(0)

args = sys.argv[2:]


def arg(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


out = Path(arg('--out')); run = out.parent
scen = json.loads((run.parent.parent / 'scenario.json').read_text())[run.name]
print(f"[fake] {role} run={run.name} scen={scen} init_adapter={arg('--init-adapter')} "
      f"dataset_start_index={arg('--dataset-start-index', '0')} lr={arg('--lr')} "
      f"cps={arg('--completions-per-step')}", flush=True)


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def checkpoint(step, partial=False):
    d = out / f'checkpoint-{step}'; d.mkdir(parents=True, exist_ok=True)
    hdr = json.dumps({'lora_A.weight': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}).encode()
    (d / 'adapter_model.safetensors').write_bytes(len(hdr).to_bytes(8, 'little') + hdr + bytes(4 if partial else 8))
    (d / 'adapter_config.json').write_text('{"r": 32, "lora_alpha": 64}\n')
    (d / 'rollout_state.json').write_text(json.dumps({'prompt_index': 64 * step, 'model_version': step + 1}))


if role == 'rank0':
    init = arg('--init-adapter')
    if init and not (Path(init) / 'adapter_model.safetensors').is_file():
        sys.exit(f'init adapter {init} missing')
    backends = [b['url'] for b in json.loads(Path('plan.json').read_text())['backends']]
    save, step = int(arg('--save-steps', 10)), 0
    if scen['rank0'] == 'stage':  # curriculum stage: run to --max-steps, save, exit 0 (like the real trainer)
        scen['fail_step'], scen['rank0'] = int(arg('--max-steps')), 'done'
    print(f"[fake] G={arg('--num-generations')} max_steps={arg('--max-steps')}", flush=True)
    while step != scen.get('fail_step'):
        step += 1
        time.sleep(scen.get('step_s', 0.3))
        for u in backends:
            try:
                post(u + '/v1/load_lora_adapter', {'lora_name': f'trl-policy-v{step}', 'lora_path': str(out)})
            except urllib.error.HTTPError as e:
                print(f'STALE ADAPTER COLLISION trl-policy-v{step} on {u}: {e}', flush=True)
                sys.exit(4)
            if step > 4:
                post(u + '/v1/unload_lora_adapter', {'lora_name': f'trl-policy-v{step - 4}'})
        print({'loss': 0.0, 'grad_norm': 0.01, 'step': step}, flush=True)
        if step % save == 0:
            checkpoint(step)
    if scen.get('partial_ckpt'):  # a save cut short by the crash
        checkpoint(step + save - step % save, partial=True)
    mode = scen['rank0']
    if mode == 'oom':
        print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB (fake)', flush=True)
    elif mode == 'rc1':
        print('fake rank 0 error, exiting 1', flush=True); sys.exit(1)
    elif mode == 'done':
        (out / 'FAKE_DONE').touch(); sys.exit(0)
    elif mode == 'sigterm_head':
        os.kill(os.getppid(), signal.SIGTERM)
    while True:
        time.sleep(60)

t0, mode = time.time(), scen.get('worker', 'follow')
while True:
    time.sleep(1)
    if mode == 'exit' and time.time() - t0 > scen.get('worker_exit_s', 5):
        print('fake node-1 rank crash, exiting 1', flush=True); sys.exit(1)
    if (out / 'FAKE_DONE').exists():
        sys.exit(0)
    if mode == 'follow' and (run / 'FAILED.json').exists():
        print('fake NCCL error: rank 0 is gone, exiting 1', flush=True); sys.exit(1)
