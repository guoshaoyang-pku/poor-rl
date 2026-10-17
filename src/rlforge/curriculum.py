"""Group-size (G) curriculum for async GRPO/GSPO: small groups first, then larger ones.

TRL's AsyncGRPOTrainer fixes `num_generations` at construction, so a curriculum runs as a chain of stages: each
stage is one trainer launch with its own G, continued from the previous stage's last checkpoint
(`--init-adapter` + `--dataset-start-index`, or `--resume-from-checkpoint` for full fine-tuning). The global batch
B (completions per optimizer update) stays fixed across stages, so the LR stays fixed too; only the number of
prompts per update changes (B / G).

Spec: comma-separated `G:updates` stages; `updates` 0 = open-ended (last stage only). Example: `8:60,16:60,32:0`.
An epoch-driven schedule can be written with `G@epochs` (e.g. `4@1,8@1,16@1,32@0`); `resolve` then needs the
dataset size and B to convert one epoch (every prompt once) into updates: ceil(rows * G / B).

    python -m rlforge.curriculum SPEC STAGE [STAGE_DONE] [--rows N --batch B]   -> JSON {G, max_steps, last, ...}
"""
from __future__ import annotations

import argparse
import json
import math


def parse(spec: str) -> list[tuple[int, str, float]]:
    """-> [(G, unit 'updates'|'epochs', amount)]; amount 0 = open-ended."""
    stages = []
    for i, part in enumerate(x.strip() for x in spec.split(',') if x.strip()):
        sep = '@' if '@' in part else ':'
        g, amount = part.split(sep)
        g, amount = int(g), float(amount)
        if g <= 0 or amount < 0:
            raise ValueError(f'bad curriculum stage {part!r}')
        stages.append((g, 'epochs' if sep == '@' else 'updates', amount))
    if not stages:
        raise ValueError('empty curriculum')
    if any(a == 0 for _, _, a in stages[:-1]):
        raise ValueError('only the last curriculum stage may be open-ended (0)')
    return stages


def resolve(spec: str, stage: int, stage_done: int = 0, rows: int | None = None, batch: int | None = None,
            open_ended_steps: int = 2000) -> dict:
    """G and the number of optimizer updates still to run in `stage` (0-based) after `stage_done` of them."""
    stages = parse(spec)
    if not 0 <= stage < len(stages):
        raise ValueError(f'stage {stage} outside {len(stages)} stages of {spec!r}')
    g, unit, amount = stages[stage]
    if amount == 0:
        total = open_ended_steps
    elif unit == 'updates':
        total = int(amount)
    else:
        if not rows or not batch:
            raise ValueError('an epoch-unit stage needs rows and batch')
        total = math.ceil(amount * rows * g / batch)
    left = total - stage_done
    if left <= 0:
        raise ValueError(f'stage {stage} already complete ({stage_done}/{total})')
    if batch and batch % g:
        raise ValueError(f'batch {batch} is not a multiple of G={g}')
    return {'G': g, 'max_steps': left, 'stage_total': total, 'stage': stage, 'stages': len(stages),
            'last': stage == len(stages) - 1, 'spec': spec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spec'); ap.add_argument('stage', type=int); ap.add_argument('stage_done', type=int, nargs='?', default=0)
    ap.add_argument('--rows', type=int); ap.add_argument('--batch', type=int)
    ap.add_argument('--open-ended-steps', type=int, default=2000)
    a = ap.parse_args()
    print(json.dumps(resolve(a.spec, a.stage, a.stage_done, a.rows, a.batch, a.open_ended_steps)))


if __name__ == '__main__':
    main()
