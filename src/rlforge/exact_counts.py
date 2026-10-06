"""Optional exact rollout token metadata for the source-pinned TRL collator."""
import ast
import hashlib
import inspect
import linecache
import textwrap

import torch

NATIVE_SHA256 = "fe0e2a7a98add7239ab99da21701eca25b8994f82b46b30c6c053766a68dac0f"


def transform(source):
    tree = ast.parse(textwrap.dedent(source))
    expected = {
        "global_n_tokens": "torch.full((self.num_processes,), float(n_trained_tokens), dtype=torch.float32)",
        "global_n_forward_tokens": "torch.full((self.num_processes,), float(n_forward_tokens), dtype=torch.float32)",
    }
    found = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in expected):
            name = node.targets[0].id
            if name in found or ast.unparse(node.value) != expected[name]:
                raise ValueError("Native rollout token counter changed")
            found[name] = node
    if found.keys() != expected.keys():
        raise ValueError("Native rollout token counters missing")
    for node in found.values():
        node.value.args[1] = node.value.args[1].args[0]
        node.value.keywords[0].value = ast.parse("torch.int64", mode="eval").body
    return ast.fix_missing_locations(tree)


def install():
    from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout
    cls = DataCollatorForRollout
    if getattr(cls, "_poor_rl_exact_counts_installed", False):
        return
    native = cls.torch_call
    source = inspect.getsource(native)
    if hashlib.sha256(source.encode()).hexdigest() != NATIVE_SHA256:
        raise ValueError("Unsupported TRL collator source for exact token counters; disable --exact-token-counts or use the tested TRL version")
    tree = transform(source)
    filename = "<poor-rl exact integer token counters>"
    patched_source = ast.unparse(tree) + chr(10)
    linecache.cache[filename] = (len(patched_source), None, patched_source.splitlines(True), filename)
    namespace = dict(native.__globals__)
    exec(compile(patched_source, filename, "exec"), namespace)
    cls.torch_call = namespace[native.__name__]
    cls._poor_rl_exact_counts_installed = True
