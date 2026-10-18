#!/usr/bin/env python3
"""Write the head plan for one Qwen3.5-4B LoRA GSPO run. Usage: make_plan.py RUN_NAME > plan.json

Choices (2026-10-17, recorded here and in the run README):
  * LoRA r32/alpha64 on all 12 Qwen3.5 projection kinds (attention, GDN, MLP) over the merged SFT ep4 model,
    frozen bf16 base, fp32 adapter; adapter-only sync through rlforge.rollout.router + lora_agent.
  * GSPO seq_mean, eps 4e-3 both sides, KL 0, no LLM judge (AIQ_HALLUC=0), G=32, max_completion 16384,
    max_staleness 3, rollout look-ahead 1 batch.
  * Batch B = completions per optimizer update = CPS_PER_RANK x trainer ranks.
    LR rule from the user: LoRA LR = 10x the full-parameter LR, scaled by sqrt(B/1024) from the 0.8B
    reference (2e-6 at B=1024)  ->  lr = 2e-5 * sqrt(B/1024).
"""
import json, math, os, sys

NAME = sys.argv[1]
NFS = '/home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017'
RUN = f'{NFS}/runs/{NAME}'
KIT = f'{NFS}/code'
PY = '/home/tione/guoshaoyang/a100_rl/venv/bin/python'
MODEL = os.environ.get('MODEL', '/home/tione/guoshaoyang/aiq_rl/models/q35_4b_sft_ep4_merged')
# Trainer nodes: first is the head (router + rank 0). Multi-node DDP runs NCCL over TCP on eth0 (no IB between containers).
TRAINER_NODES = os.environ.get('TRAINER_NODES', '172.16.0.116').split(',')
TRAINER_RANKS = 8 * len(TRAINER_NODES)
CPS_PER_RANK = int(os.environ.get('CPS_PER_RANK', 256))
B = CPS_PER_RANK * TRAINER_RANKS
# G curriculum (user, 2026-10-17): e.g. CURRICULUM="8:60,16:60,32:0" = G8 for 60 updates, G16 for 60, then G32
# open-ended. Each stage is its own run; STAGE picks the stage, STAGE_DONE = updates of this stage already done
# (a mid-stage resume). B, and therefore LR, stay fixed across stages.
CURRICULUM = os.environ.get('CURRICULUM', '')
STAGE, STAGE_DONE = int(os.environ.get('STAGE', 0)), int(os.environ.get('STAGE_DONE', 0))
if CURRICULUM:
    _here = os.path.dirname(os.path.abspath(__file__))  # kit layout: ./src ; poor-rl layout: ../../src
    sys.path[:0] = [p for p in (os.path.join(_here, 'src'), os.path.join(_here, '..', '..', 'src')) if os.path.isdir(p)]
    from rlforge.curriculum import resolve
    _c = resolve(CURRICULUM, STAGE, STAGE_DONE, batch=B, open_ended_steps=int(os.environ.get('OPEN_ENDED_STEPS', 2000)))  # not MAX_STEPS: the watcher passes the previous stage's
    NGEN, MAX_STEPS = _c['G'], str(_c['max_steps'])
else:
    NGEN, MAX_STEPS = int(os.environ.get('NGEN', 32)), os.environ.get('MAX_STEPS', '2000')
assert CPS_PER_RANK % NGEN == 0, (CPS_PER_RANK, NGEN)
LR = 2e-5 * math.sqrt(B / 1024)
ALL_ROLLOUT = {  # ip: (gpus, base_port)
    '172.16.0.105': ('0,1,2,3,4,5,6,7', 8200),   # a100_t1_2
    '172.16.0.48': ('0,1,2,3,4,5,6,7', 8200),    # a100_t1_4
    '172.16.0.15': ('0,1,2,3,4,5', 8200),        # a100_t1_3
    '172.16.0.123': ('0,1,2,3,4,5,6,7', 8200),   # a100_perm
}
HOSTS = os.environ.get('ROLLOUT_HOSTS')  # comma IPs; default every non-trainer host (t1_3 can be taken by others)
ROLLOUT = {ip: v for ip, v in ALL_ROLLOUT.items() if ip not in TRAINER_NODES and (not HOSTS or ip in HOSTS.split(','))}
backends = []
for ip, (gpus, base) in ROLLOUT.items():
    for i, _ in enumerate(gpus.split(',')):
        backends.append({'url': f'http://{ip}:{base + i}', 'agent': f'http://{ip}:8399'})
targets = 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj'
def launch(rank):
    multi = ['--num_machines', str(len(TRAINER_NODES)), '--machine_rank', str(rank), '--main_process_ip',
             TRAINER_NODES[0], '--main_process_port', '29655'] if len(TRAINER_NODES) > 1 else []
    return [PY, '-m', 'accelerate.commands.launch', '--num_processes', str(TRAINER_RANKS), *multi,
            '--mixed_precision', 'bf16', '--dynamo_backend', 'no', '-m', 'rlforge.trainer']


trainer = ['--model', MODEL, '--server-url', f'http://{TRAINER_NODES[0]}:8300', '--train', os.environ.get('DATA', f'{NFS}/data/train_think.jsonl'),
           '--out', f'{RUN}/trainer_out', '--epochs', '10', '--lr', f'{LR:.4g}',
           '--completions-per-step', str(CPS_PER_RANK), '--num-generations', str(NGEN), '--max-completion', '16384',
           '--max-steps', MAX_STEPS, '--save-steps', os.environ.get('SAVE_STEPS', '10'),
           '--max-staleness', '3', '--rollout-ahead-batches', os.environ.get('AHEAD', '1'), '--max-inflight-tasks', str((1 + int(os.environ.get('AHEAD', '1'))) * B),
           '--report-to', 'none', '--reward', 'aiq_think_reward_v4:think_reward', '--request-timeout', '5400',
           '--gspo', '--gspo-norm', 'seq_mean', '--gspo-eps-low', '4e-3', '--gspo-eps-high', '4e-3',
           '--per-seq-forward', 'auto', '--kl-beta', '0', '--prefix-share', 'on', '--token-budget', '0',
           '--dp-route', 'off', '--queue-maxsize', str((1 + int(os.environ.get('AHEAD', '1'))) * B),  # 512 < B throttled rollout via score backpressure (r6)
           '--score-concurrency', '32', '--judged-max-staleness', '-1',
           '--drop-audit', 'on', '--subbatch-tokens', os.environ.get('SB_TOKENS', '32768'), '--sb-ckpt', 'auto',
           '--balance-rows', 'on', '--dtype', 'bfloat16',
           '--lora', '--lora-r', '32', '--lora-alpha', '64', '--lora-target-modules', targets]
if os.environ.get('INIT_ADAPTER'):  # continue from a weights-only LoRA checkpoint (see trainer --init-adapter)
    trainer += ['--init-adapter', os.environ['INIT_ADAPTER'], '--dataset-start-index', os.environ.get('DATA_START', '0')]
plan = {
    'run_dir': RUN, 'python': PY, 'router_port': 8300, 'backends': backends,
    'min_healthy': int(os.environ.get('MIN_HEALTHY', 16)), 'backend_wait_s': 1800,
    'batch_completions_per_update': B, 'lr': LR,
    'resume': {'init_adapter': os.environ.get('INIT_ADAPTER'), 'dataset_start_index': int(os.environ.get('DATA_START', 0)),
               'prior_updates': int(os.environ.get('PRIOR_UPDATES', 0))},
    'curriculum': ({'schedule': CURRICULUM, 'stage': STAGE, 'stage_done': STAGE_DONE, 'G': NGEN} if CURRICULUM else None),
    'lr_rule': '2e-5*sqrt(B/1024) (LoRA = 10x full-param 2e-6@1024)',
    'env': {'PYTHONPATH': f'{KIT}/src:{NFS}/data', 'AIQ_HALLUC': '0', 'RLFORGE_FAST_LOGPROB': '1',
            'RLFORGE_LOGPROB_BWD': 'tf32', 'RLFORGE_SB_ACT_GB': os.environ.get('SB_ACT_GB', '16'),
            'RLFORGE_SB_MEM_FULL_KB': '420', 'RLFORGE_SB_MEM_CKPT_KB': '16', 'RLFORGE_SB_MEM_BASE_KB': '53',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '8',
            'RLFORGE_TASK_LOG': f'{RUN}/task_split.jsonl', 'AIQ_SAMPLE_LOG': f'{RUN}/rollout_samples.jsonl',
            'RLFORGE_DROP_AUDIT_PATH': f'{RUN}/drop_audit.jsonl',
            'RLFORGE_DROP_ZERO_VAR': os.environ.get('DROP_ZERO_VAR', '1'),
            'NCCL_SOCKET_IFNAME': 'eth0', 'GLOO_SOCKET_IFNAME': 'eth0', 'NCCL_IB_DISABLE': '1'},
    'trainer_env': {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
                    'ACCELERATE_MIXED_PRECISION': 'bf16'},
    'trainer_argv': launch(0) + trainer,
    'trainer_nodes': TRAINER_NODES,
    'worker_argvs': {ip: launch(r) + trainer for r, ip in enumerate(TRAINER_NODES) if r > 0},
    'stall_s': 3600,
}
print(json.dumps(plan, indent=2))
