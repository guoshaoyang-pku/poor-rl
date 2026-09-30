"""Reward function protocol and loader.

A reward function is any callable with the TRL signature::

    fn(completions, prompts, completion_ids, answer, cap: int, **dataset_columns) -> list[float]

Load a custom one with ``module.submodule:function`` (the module must be importable,
e.g. on PYTHONPATH)::

    python -m rlforge.trainer --reward my_project.rewards:my_reward_fn ...

The built-in MCQ/ranking reward is ``rlforge.rewards.mcq:mcq_reward``.
"""
from __future__ import annotations

import importlib


def load_reward_fn(spec: str):
    if ":" not in spec:
        raise ValueError(f"reward spec must be 'module:function', got {spec!r}")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise ValueError(f"{spec!r} does not resolve to a callable")
    return fn
