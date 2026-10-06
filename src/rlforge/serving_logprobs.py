"""Exact native sampled-token text cache and flat final-only logprob storage."""
from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

_RENDER_COUNTS = dict(fast=0, fallback=0)

_FILES = {
    "v1/engine/logprobs.py": "14bbe78d92a2c451d79b3d9dd9d5268bc57e16bad5079e44ce9ed2cd805e873e",
    "tokenizers/detokenizer_utils.py": "e17ef23dae7bbb1db5875024c249567b988d8611b560d095c3f786d1e697fe11",
    "logprobs.py": "3cc2ed6cdd30572d29f9b293ddbc14ad318c6b5d5dc7fdd36bee2f8af259b373",
    "entrypoints/openai/completion/protocol.py": "5f7930635fd3924826015211032fb298206b27c7f04cd6cd2224bb128e4a3a50",
    "entrypoints/openai/completion/serving.py": "75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527",
    "tokenizers/hf.py": "53b0dd497817c873d0ef02a7b716affff836c344f85947a4c59b61e7acffa95d",
}
_INSTALLED = False
_ENABLED = False
_CACHE_LIMIT = 16384
_COUNTS = dict(cache_hits=0, cache_misses=0, tokenizer_fallback=0,
               utf8_context=0, context_noop=0, flat_requests=0, native_requests=0)


def _cached_text(native, tokenizer, token_ids):
    tokenizer_type = type(tokenizer)
    if (tokenizer_type.__module__ != "vllm.tokenizers.hf"
            or tokenizer_type.__name__ not in
            ("TokenizerPoolCachedQwen2Tokenizer", "CachedQwen2Tokenizer")):
        _COUNTS["tokenizer_fallback"] += 1
        return native(tokenizer, token_ids)
    signature = (len(tokenizer), tokenizer.clean_up_tokenization_spaces,
                 id(tokenizer.backend_tokenizer))
    state = getattr(tokenizer, "_poor_rl_native_sample_text", None)
    if state is None or state[0] != signature:
        state = (signature, OrderedDict())
        tokenizer._poor_rl_native_sample_text = state
    cache = state[1]
    result = []
    for token in token_ids:
        if token in cache:
            text = cache[token]
            cache.move_to_end(token)
            _COUNTS["cache_hits"] += 1
        else:
            text = native(tokenizer, [token])[0]
            _COUNTS["cache_misses"] += 1
            if 0 <= token < signature[0]:
                cache[token] = text
                if len(cache) > _CACHE_LIMIT:
                    cache.popitem(last=False)
        result.append(text)
    return result



def _install_renderer():
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.entrypoints.openai.completion.protocol import CompletionLogProbs
    from vllm.logprobs import FlatLogprobs, Logprob
    native = OpenAIServingCompletion._create_completion_logprobs

    def create(self, token_ids, top_logprobs, num_output_top_logprobs, tokenizer,
               logprob_token_ids=None, initial_text_offset=0, return_as_token_id=None):
        use_placeholders = (return_as_token_id if return_as_token_id is not None
                            else self.return_tokens_as_token_ids)
        flat = type(top_logprobs) is FlatLogprobs
        as_list = type(top_logprobs) is list
        supported = (_ENABLED and (flat or as_list)
                     and type(token_ids) in (list, tuple)
                     and num_output_top_logprobs == 0 and not logprob_token_ids
                     and not use_placeholders)
        if supported:
            n = len(token_ids)
            fields = (top_logprobs.start_indices, top_logprobs.end_indices,
                      top_logprobs.token_ids, top_logprobs.logprobs,
                      top_logprobs.decoded_tokens, top_logprobs.ranks) if flat else (top_logprobs,)
            supported = all(type(field) is list and len(field) == n for field in fields)
        if supported:
            offsets, selected, words, top = [], [], [], []
            offset = initial_text_offset
            for index, token in enumerate(token_ids):
                if flat:
                    if (top_logprobs.start_indices[index] != index
                            or top_logprobs.end_indices[index] != index + 1
                            or top_logprobs.token_ids[index] != token):
                        supported = False
                        break
                    logprob = top_logprobs.logprobs[index]
                    word = top_logprobs.decoded_tokens[index]
                else:
                    item = top_logprobs[index]
                    if type(item) is not dict or len(item) != 1:
                        supported = False
                        break
                    value = item.get(token)
                    if type(value) is not Logprob:
                        supported = False
                        break
                    logprob, word = value.logprob, value.decoded_token
                if type(word) is not str or type(logprob) is not float:
                    supported = False
                    break
                value = max(logprob, -9999.0)
                offsets.append(offset)
                selected.append(value)
                words.append(word)
                top.append({word: value})
                offset += len(word)
            if supported:
                result = CompletionLogProbs(text_offset=offsets, token_logprobs=selected,
                                           tokens=words, top_logprobs=top)
                _RENDER_COUNTS['fast'] += 1
                return result
        _RENDER_COUNTS['fallback'] += 1
        result = native(self, token_ids, top_logprobs, num_output_top_logprobs, tokenizer,
                        logprob_token_ids, initial_text_offset, return_as_token_id)
        return result

    OpenAIServingCompletion._create_completion_logprobs = create


def install():
    global _INSTALLED
    if _INSTALLED:
        return stats()
    import vllm
    from vllm.v1.engine import logprobs as engine
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.sampling_params import RequestOutputKind

    root = Path(vllm.__file__).parent
    for name, expected in _FILES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Native sampled-logprob source changed: {name}")
    native_update = engine.LogprobsProcessor._update_sample_logprobs
    native_convert = engine.convert_ids_list_to_tokens
    native_context = engine.LogprobsProcessor._get_sampled_context_ids
    native_sampling = CompletionRequest.to_sampling_params

    def convert(tokenizer, token_ids):
        if _ENABLED:
            return _cached_text(native_convert, tokenizer, token_ids)
        return native_convert(tokenizer, token_ids)

    def context(source, decoded):
        if not _ENABLED or any(text.endswith("�") for text in decoded):
            _COUNTS["utf8_context"] += 1
            return native_context(source)
        _COUNTS["context_noop"] += 1
        return []

    tree = ast.parse(textwrap.dedent(inspect.getsource(native_update)))
    changes = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == "context_token_ids"
                        for target in node.targets)):
            node.value = ast.parse(
                "poor_rl_context(self.logprobs, decoded_tokens_list)", mode="eval"
            ).body
            changes += 1
    if changes != 1:
        raise RuntimeError("Native sampled-logprob context assignment changed")
    namespace = dict(native_update.__globals__,
                     convert_ids_list_to_tokens=convert, poor_rl_context=context)
    exec(compile(ast.fix_missing_locations(tree),
                 "<native sampled-logprob cached text>", "exec"), namespace)

    def sampling(self, max_tokens, default_sampling_params=None):
        params = native_sampling(self, max_tokens, default_sampling_params)
        supported = (_ENABLED and self.logprobs == 0 and not self.stream
                     and not self.echo and not self.logprob_token_ids
                     and params.prompt_logprobs is None
                     and params.output_kind == RequestOutputKind.FINAL_ONLY)
        if supported:
            params.flat_logprobs = True
            _COUNTS["flat_requests"] += 1
        else:
            _COUNTS["native_requests"] += 1
        return params

    _install_renderer()
    engine.LogprobsProcessor._update_sample_logprobs = namespace["_update_sample_logprobs"]
    CompletionRequest.to_sampling_params = sampling
    _INSTALLED = True
    return stats()


def toggle(enabled):
    global _ENABLED
    _ENABLED = enabled
    return stats()


def stats():
    return dict(pid=os.getpid(), installed=_INSTALLED, enabled=_ENABLED,
                counts=dict(_COUNTS), rendering=dict(_RENDER_COUNTS), cache_limit=_CACHE_LIMIT, native_files=_FILES,
                scope="Native isolated-token text only, UTF8 correction and cumulative LP unchanged; native FlatLogprobs final-only logprobs0. Native detokenizer/stops/full fields/metrics/RNG retained; static installed Qwen tokenizer supported.")


def register():
    if os.environ.get("RLFORGE_SERVING_LOGPROBS_CACHE", "0") == "1":
        install()
        toggle(True)
