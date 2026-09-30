#!/usr/bin/env python3
"""Reward/parser tests for rlforge.rewards.mcq.

Pins the semantics that training and eval share: letter parsing, ranking parsing
(both separators), concordance scoring, unparseable/truncation paths, and the
per-task accounting counters.
"""
import os

from rlforge.rewards import load_reward_fn
from rlforge.rewards.mcq import (
    TRUNCATED_REWARD,
    UNPARSED_REWARD,
    is_ranking_gold,
    mcq_reward,
    parse_answer,
    parse_order,
    ranking_score,
)


def test_parse_answer_last_tag_wins():
    assert parse_answer("reasoning <answer>b</answer> more <answer>C</answer>") == "C"
    assert parse_answer("no tag at all") is None
    # a ranking chain in the tag must NOT be greedily read as the letter A
    assert parse_answer("<answer>A<B<C<D<E</answer>") is None


def test_parse_order_both_separators():
    assert parse_order("<answer>A<B<C<D<E</answer>") == ("A", "B", "C", "D", "E")
    assert parse_order("<answer>E>D>C>B>A</answer>") == ("E", "D", "C", "B", "A")
    assert parse_order("<answer>A</answer>") is None
    assert parse_order("<answer>A<B<C<D</answer>") is None  # 4 letters, not 5


def test_is_ranking_gold_shape_based():
    assert is_ranking_gold("A<B<C<D<E")
    assert is_ranking_gold("E>D>C>B>A")
    assert not is_ranking_gold("B")
    assert not is_ranking_gold(None)


def test_ranking_score_scale():
    assert ranking_score(("A", "B", "C", "D", "E"), "A<B<C<D<E") == 1.0
    assert ranking_score(("E", "D", "C", "B", "A"), "A<B<C<D<E") == -1.0
    # one adjacent swap: 9/10 concordant -> 9/5 - 1 = 0.8
    assert abs(ranking_score(("B", "A", "C", "D", "E"), "A<B<C<D<E") - 0.8) < 1e-9
    assert ranking_score(None, "A<B<C<D<E") == UNPARSED_REWARD


def test_mcq_reward_paths_and_counters(tmp_path, monkeypatch):
    monkeypatch.setenv("RLFORGE_TASK_LOG", str(tmp_path / "split.jsonl"))
    completions = [
        "think <answer>C</answer>",          # correct
        "think <answer>A</answer>",          # wrong
        "no tag",                            # unparseable
        "loop loop loop",                    # truncated (ids >= cap)
        "<answer>A<B<C<D<E</answer>",        # ranking exact
    ]
    ids = [[1, 2], [1, 2], [1, 2], list(range(10)), [1, 2]]
    answers = ["C", "C", "C", "C", "A<B<C<D<E"]
    src = ["s1", "s1", "s1", "s1", "loss_ranked"]
    rs = mcq_reward(completions, ["p"] * 5, ids, answers, cap=10, source=src)
    assert rs == [1.0, 0.0, UNPARSED_REWARD, TRUNCATED_REWARD, 1.0]

    import json
    line = json.loads(open(tmp_path / "split.jsonl").read().strip())
    assert line["n_mcq"] == 4 and line["n_rank"] == 1
    assert line["mcq_correct"] == 1 and line["mcq_unparsed"] == 1 and line["mcq_trunc"] == 1
    assert line["rank_exact"] == 1
    assert line["by_source"]["s1"][0] == 4
    assert line["by_source"]["loss_ranked"][2] == 1


def test_reward_loader():
    fn = load_reward_fn("rlforge.rewards.mcq:mcq_reward")
    assert callable(fn)
