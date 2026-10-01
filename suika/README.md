# suika (合成大西瓜) codebase branch

The Suika game-RL codebase lives in this branch as a self-contained subtree:

- `rl/` — distributed DQN / Qwen-BC / expert-iteration stack (learner + actors +
  evaluator + inference servers, configs, plotting and pull scripts).
- `engine/` — the physics engine and game rules (`part2/suika_env.py`: tempo
  cadence, double-watermelon merge-disappear rule, C settlement scan).
- `docs/` — design document (`RL设计与分析.md`), baseline diagnostics, and the
  full experiment log.

Trial curves and the experiment registry for these runs are tracked on `main`
under `examples/suika_trials/` and rendered by the algorithm-neutral panel
(`python -m rlforge.dashboard --rl-data-root examples/suika_trials`).

Hostnames and absolute node paths are sanitized to `node_a..node_f` and
`/path/to/...` placeholders; reproduce by editing `rl/configs/*.yaml`.
