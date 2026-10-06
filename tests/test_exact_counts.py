import ast
import hashlib
import inspect
import textwrap

import pytest
import torch

from rlforge import exact_counts


def test_native_collator_preserves_batch_and_metadata(monkeypatch):
    pytest.importorskip("trl")
    from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout

    native = DataCollatorForRollout.torch_call
    source = inspect.getsource(native)
    if hashlib.sha256(source.encode()).hexdigest() != exact_counts.NATIVE_SHA256:
        pytest.skip("Requires the recorded TRL collator source")
    monkeypatch.setattr(DataCollatorForRollout, "torch_call", native)
    monkeypatch.setattr(DataCollatorForRollout, "_poor_rl_exact_counts_installed", False, raising=False)
    groups = [[dict(input_ids=[1, 2, 3 + index], completion_mask=[0, 1, 1],
                   old_log_probs=[0., -.2, -.3], advantage=(index - 1) / 7.,
                   group_id=index, metrics={"reward": .2 + index / 10})] for index in range(4)]
    old_collator, new_collator = [DataCollatorForRollout(99, 4) for _ in range(2)]
    old = native(old_collator, [groups])
    exact_counts.install()
    installed = DataCollatorForRollout.torch_call
    exact_counts.install()
    assert installed is DataCollatorForRollout.torch_call
    new = new_collator.torch_call([groups])
    assert old.keys() == new.keys()
    for key in old:
        assert torch.equal(old[key], new[key])
        assert old[key].shape == new[key].shape
        expected = torch.int64 if key in ("global_n_tokens", "global_n_forward_tokens") else old[key].dtype
        assert new[key].dtype == expected
    assert old_collator.groups_trained == new_collator.groups_trained
    assert old_collator.metrics == new_collator.metrics

    def counter_ast(tree):
        return ast.Module(body=[node for node in tree.body[0].body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("global_n_tokens", "global_n_forward_tokens")], type_ignores=[])

    for count in (2**24 + 1, 18_632_469, 2**25 + 3, 2**53 + 1):
        old_ns = dict(torch=torch, self=old_collator, n_trained_tokens=count, n_forward_tokens=count + 2)
        new_ns = dict(old_ns)
        exec(compile(counter_ast(ast.parse(textwrap.dedent(source))), "native counters", "exec"), old_ns)
        exec(compile(counter_ast(exact_counts.transform(source)), "exact counters", "exec"), new_ns)
        assert new_ns["global_n_tokens"].tolist() == [count] * 4
        assert new_ns["global_n_forward_tokens"].tolist() == [count + 2] * 4
        assert torch.equal(new_ns["global_n_tokens"].float(), old_ns["global_n_tokens"])


def test_source_mismatch_fails_before_patch(monkeypatch):
    pytest.importorskip("trl")
    from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout

    native = DataCollatorForRollout.torch_call
    monkeypatch.setattr(DataCollatorForRollout, "_poor_rl_exact_counts_installed", False, raising=False)
    monkeypatch.setattr(exact_counts, "NATIVE_SHA256", "unsupported")
    with pytest.raises(ValueError, match="Unsupported TRL collator"):
        exact_counts.install()
    assert DataCollatorForRollout.torch_call is native


def test_changed_counter_assignment_is_rejected():
    with pytest.raises(ValueError, match="counter"):
        exact_counts.transform("def torch_call(self):\n    global_n_tokens = torch.ones(1)\n")
