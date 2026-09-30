"""Built-in MCQ / ranking reward with per-task and per-source accounting.

Answer contract: a closing `<answer>X</answer>` tag (MCQ letter) or a chained order
(`<answer>A<B<C<D<E</answer>`, best-first; `<` means "left has lower held-out loss").

Reward semantics (shared by training and eval -- do not fork them):
  MCQ     : +1 correct letter / 0 wrong / -0.5 unparseable / -2 truncated at the cap
  ranking : concordance in [-1, +1] (exact +1, coin flip 0, full reversal -1),
            -0.5 unparseable, -2 truncated

The separator only states direction to the reader; the letter sequence is always
best-first, so `>` and `<` parse identically.
"""
from __future__ import annotations

import json
import os
import re
import time

# A single-letter answer never contains `<`, so the letter path blocks it: without that,
# the final tag of a ranking completion would be greedily re-read as `A`.
ANSWER_RE = re.compile(r"<answer>\s*([^<]*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
# A ranked answer is a chain of letters, so its tag body contains `<`, which ANSWER_RE's
# `[^<]` class can never swallow -- ranking answers need their own pattern.
ORDER_TAG_RE = re.compile(
    r"<answer>\s*([A-Ea-e](?:\s*[<>]\s*[A-Ea-e])+)\s*</answer>", re.DOTALL | re.IGNORECASE
)
LETTER_RE = re.compile(r"^[^A-Za-z]*([A-E])\b")
SEPARATOR_RE = re.compile(r"\s*[<>]\s*")

UNPARSED_REWARD = -0.5
TRUNCATED_REWARD = -2.0

N_RANK_LETTERS = 5


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return "".join(m.get("content", "") for m in completion)
    return str(completion)


def parse_answer(text: str):
    tags = ANSWER_RE.findall(text)
    if not tags:
        return None
    m = LETTER_RE.match(tags[-1])
    return m.group(1).upper() if m else None


def _as_letters(value):
    """Accept `None`, a chain string (`A<B<C<D<E` / `A>B>..`) or a letter sequence."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.upper() for part in SEPARATOR_RE.split(value.strip()) if part]
    else:
        parts = [str(part).strip().upper() for part in value]
    return tuple(parts) or None


def parse_order(text: str):
    """Final ranked answer -> letters best-first, or None."""
    tags = ORDER_TAG_RE.findall(text)
    if not tags:
        return None
    letters = _as_letters(tags[-1])
    ok = letters and len(letters) == N_RANK_LETTERS and set(letters) == set("ABCDE")
    return letters if ok else None


def is_ranking_gold(gold: str) -> bool:
    """True when the gold is a ranked chain rather than a single letter.

    Test the shape (more than one letter), not one specific separator: keys may move
    between `>` and `<`, and a `">" in gold` test silently stops recognising ranking
    questions when they do.
    """
    letters = _as_letters(gold)
    return bool(letters) and len(letters) > 1


def ranking_score(prediction, gold: str) -> float:
    """Concordance score in [-1, +1]: +1 for the exact order, -1 for the full reversal.

    `concordant` counts the choice pairs the prediction orders the same way as the gold;
    there are C(5,2) = 10 of them, so 5 concordant pairs (a coin flip) scores 0.
    """
    expected = _as_letters(gold)
    if expected is None or len(expected) != N_RANK_LETTERS or set(expected) != set("ABCDE"):
        raise ValueError(f"Invalid gold ranking: {gold!r}")
    predicted = _as_letters(prediction)
    if predicted is None or len(predicted) != N_RANK_LETTERS or set(predicted) != set("ABCDE"):
        return UNPARSED_REWARD
    if predicted == expected:
        return 1.0
    positions = {letter: index for index, letter in enumerate(expected)}
    concordant = sum(
        positions[predicted[left]] < positions[predicted[right]]
        for left in range(N_RANK_LETTERS)
        for right in range(left + 1, N_RANK_LETTERS)
    )
    return concordant / 5.0 - 1.0


def _log_task_split(stats: dict) -> None:
    """Append one JSON line per reward call when RLFORGE_TASK_LOG (or legacy
    AIQ_TASK_LOG) is set. Best-effort: a logging failure must never change a reward.

    The trainer logs a single reward column, which mixes task types in a mixed pool.
    Without this the mean can only be read as one number and a task that stops parsing
    looks like a task that got worse.
    """
    path = os.environ.get("RLFORGE_TASK_LOG") or os.environ.get("AIQ_TASK_LOG", "")
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(stats, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _gold_probe(gold, kwargs: dict) -> dict:
    """Record what the reward actually received: `answer` may not arrive as the dataset
    row's value, and an unexpected binding is invisible from the reward's own output."""
    return {
        "answer_type": type(gold).__name__,
        "answer_repr": repr(gold)[:60],
        "answer_is_ranking": is_ranking_gold(gold),
        "kw_task": repr(kwargs.get("task"))[:30],
        "kw_choices": repr(kwargs.get("num_choices"))[:12],
    }


def mcq_reward(completions, prompts, completion_ids, answer, cap=16384, **kwargs):
    """TRL reward function. Module-level so the rollout child process can pickle it.

    Dataset contract: each row has `prompt` (chat list or str) and `answer` (letter or
    chain). Optional `source` column enables per-source accounting in the task log.
    """
    rewards = []
    split = {
        "t": round(time.time(), 1),
        "n_mcq": 0, "n_rank": 0,
        "mcq_correct": 0, "mcq_unparsed": 0, "mcq_trunc": 0,
        "rank_exact": 0, "rank_unparsed": 0, "rank_trunc": 0, "rank_score_sum": 0.0,
        "by_source": {},
    }
    if len(answer):
        split.update(_gold_probe(answer[0], kwargs))
    src_col = kwargs.get("source") or ["?"] * len(completions)
    for comp, ids, gold, src in zip(completions, completion_ids, answer, src_col):
        ranking = is_ranking_gold(gold)
        split["n_rank" if ranking else "n_mcq"] += 1
        bs = split["by_source"].setdefault(str(src), [0, 0.0, 0, 0])
        bs[0] += 1
        if len(ids) >= cap:  # hit the token cap -> truncated
            r = TRUNCATED_REWARD
            split["rank_trunc" if ranking else "mcq_trunc"] += 1
        elif ranking:
            r = ranking_score(parse_order(_completion_text(comp)), gold.strip())
            split["rank_score_sum"] += r
            split["rank_exact"] += int(r == 1.0)
            split["rank_unparsed"] += int(r == UNPARSED_REWARD)
        else:
            pred = parse_answer(_completion_text(comp))
            if pred is None:
                r = UNPARSED_REWARD
                split["mcq_unparsed"] += 1
            elif pred == gold.strip().upper():
                r = 1.0
                split["mcq_correct"] += 1
            else:
                r = 0.0
        rewards.append(r)
        bs[1] = round(bs[1] + r, 4)
        bs[2] += int(r == 1.0)
        bs[3] += int(r == TRUNCATED_REWARD)
    split["rank_score_sum"] = round(split["rank_score_sum"], 4)
    _log_task_split(split)
    return rewards
