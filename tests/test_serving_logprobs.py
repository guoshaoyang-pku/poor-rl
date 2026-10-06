"""Check the optional native vLLM sampled-logprob cache against native fields."""

import importlib
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="Native vLLM gates run in the serving environment")

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.hf import maybe_make_thread_pool
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsLists

from rlforge import serving_logprobs


@pytest.fixture
def installed_cache(monkeypatch):
    import vllm.v1.engine.logprobs as engine

    module = importlib.reload(serving_logprobs)
    native_update = engine.LogprobsProcessor._update_sample_logprobs
    native_sampling = CompletionRequest.to_sampling_params
    native_render = OpenAIServingCompletion._create_completion_logprobs
    monkeypatch.setattr(engine.LogprobsProcessor, "_update_sample_logprobs", native_update)
    monkeypatch.setattr(CompletionRequest, "to_sampling_params", native_sampling)
    monkeypatch.setattr(OpenAIServingCompletion, "_create_completion_logprobs", native_render)
    module.install()
    module.toggle(True)
    yield module, native_update, native_sampling, native_render
    module.toggle(False)


def test_native_final_fields_and_utf8_are_exact(installed_cache):
    import numpy as np

    path = os.environ.get("RLFORGE_TEST_TOKENIZER_PATH")
    if not path:
        pytest.skip("Set RLFORGE_TEST_TOKENIZER_PATH to a local Qwen tokenizer")
    module, native_update, native_sampling, native_render = installed_cache
    tokenizer = maybe_make_thread_pool(get_tokenizer(path, local_files_only=True))
    request = CompletionRequest(model="test", prompt="test", logprobs=0)
    native_params = native_sampling(request, 256, {})
    cached_params = request.to_sampling_params(256, {})
    assert not native_params.flat_logprobs and cached_params.flat_logprobs
    renderer = object.__new__(OpenAIServingCompletion)
    renderer.return_tokens_as_token_ids = False
    for text in ["中文𠮷 🌈 English  spaces\n\t<|im_end|>", "hello world"]:
        ids = tokenizer.encode(text)
        native = LogprobsProcessor.from_new_request(
            tokenizer, SimpleNamespace(sampling_params=native_params))
        cached = LogprobsProcessor.from_new_request(
            tokenizer, SimpleNamespace(sampling_params=cached_params))
        for index, token in enumerate(ids):
            value = LogprobsLists(
                np.asarray([[token]], dtype=np.int32),
                np.asarray([[-0.01 - index * 0.001]], dtype=np.float32),
                np.asarray([1 + index % 10], dtype=np.int32), None)
            native_update(native, value)
            cached._update_sample_logprobs(value)
        assert native.cumulative_logprob == cached.cumulative_logprob
        old = native_render(renderer, ids, native.logprobs, 0, tokenizer)
        new = renderer._create_completion_logprobs(ids, cached.logprobs, 0, tokenizer)
        assert old.model_dump() == new.model_dump()
    assert module.stats()["rendering"]["fast"] == 2


@pytest.mark.parametrize("changes", [
    {"stream": True}, {"logprobs": 2}, {"echo": True},
    {"logprob_token_ids": [1]}, {"logprobs": None},
])
def test_unsupported_requests_use_native_storage(installed_cache, changes):
    module, _, native_sampling, _ = installed_cache
    fields = dict(model="test", prompt="test", logprobs=0)
    fields.update(changes)
    request = CompletionRequest(**fields)
    assert request.to_sampling_params(256, {}).flat_logprobs == native_sampling(
        request, 256, {}).flat_logprobs
    module.toggle(False)
    request = CompletionRequest(model="test", prompt="test", logprobs=0)
    assert not request.to_sampling_params(256, {}).flat_logprobs


def test_source_mismatch_does_not_patch_native(monkeypatch):
    module = importlib.reload(serving_logprobs)
    native = LogprobsProcessor._update_sample_logprobs
    monkeypatch.setitem(module._FILES, "v1/engine/logprobs.py", "0" * 64)
    with pytest.raises(RuntimeError, match="source changed"):
        module.install()
    assert LogprobsProcessor._update_sample_logprobs is native
