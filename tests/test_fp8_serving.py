"""Numerical and loading checks for the optional H200 vLLM FP8 plugin."""

import pytest
import torch
from types import SimpleNamespace

pytest.importorskip("vllm", reason="vLLM serving tests run in the H200 environment")

from rlforge import fp8_serving
from vllm.model_executor.layers.quantization.online.fp8 import (
    OnlineLinearBase, _fp8_channel_scale, _fp8_quant_per_channel,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, UnquantizedEmbeddingMethod, VocabParallelEmbedding,
)


def _vocab_layer(cls, method):
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    layer = object.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.quant_method = method
    method.create_weights(layer, 64, [32], 64, 32, torch.bfloat16,
                          weight_loader=default_weight_loader)
    return layer


def test_fused_projections_preserve_per_row_quantization():
    generator = torch.Generator().manual_seed(7)
    qkv = torch.randn(48, 64, generator=generator, dtype=torch.float32)
    scale = _fp8_channel_scale(qkv.abs().amax(dim=-1, keepdim=True))
    quantized = _fp8_quant_per_channel(qkv, scale)
    pieces = []
    scales = []
    for piece in qkv.split(16):
        piece_scale = _fp8_channel_scale(piece.abs().amax(dim=-1, keepdim=True))
        pieces.append(_fp8_quant_per_channel(piece, piece_scale))
        scales.append(piece_scale)
    assert torch.equal(quantized.view(torch.uint8), torch.cat(pieces).view(torch.uint8))
    assert torch.equal(scale, torch.cat(scales))


def test_fp32_master_quantization_avoids_bf16_intermediate():
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(64, 256, generator=generator, dtype=torch.float32)
    scale = _fp8_channel_scale(weight.abs().amax(dim=-1, keepdim=True))
    quantized = _fp8_quant_per_channel(weight, scale)
    rounded = weight.bfloat16().float()
    rounded_scale = _fp8_channel_scale(rounded.abs().amax(dim=-1, keepdim=True))
    rounded_quantized = _fp8_quant_per_channel(rounded, rounded_scale)
    assert not torch.equal(quantized.view(torch.uint8), rounded_quantized.view(torch.uint8))


def test_reload_metadata_loads_fp32_and_output_stays_bf16(monkeypatch):
    monkeypatch.setenv("RLFORGE_FP8_FORWARD", "native")
    received = {}

    def create_weights(self, layer, input_partition, output_partitions,
                       input_size, output_size, params_dtype, **attrs):
        received["load_dtype"] = params_dtype
        layer.weight = torch.nn.Parameter(
            torch.empty(sum(output_partitions), input_partition, dtype=params_dtype),
        )

    def init_kernel(**kwargs):
        received["kernel"] = kwargs
        return object()

    monkeypatch.setattr(OnlineLinearBase, "create_weights", create_weights)
    monkeypatch.setattr(fp8_serving, "get_tensor_model_parallel_world_size", lambda: 1)
    # The superclass selects a backend before this method forces CUTLASS.
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.fp8.init_fp8_linear_kernel",
        init_kernel,
    )
    monkeypatch.setattr(fp8_serving, "init_fp8_linear_kernel", init_kernel)
    method = object.__new__(fp8_serving.MasterFP8LinearMethod)
    method.input_dtype = method.out_dtype = torch.bfloat16
    method.create_weights(torch.nn.Module(), 64, [32], 64, 32, torch.bfloat16)
    assert received["load_dtype"] == torch.float32
    assert received["kernel"]["input_dtype"] == torch.bfloat16
    assert received["kernel"]["out_dtype"] == torch.bfloat16
    assert received["kernel"]["force_kernel"] is fp8_serving.CutlassFP8ScaledMMLinearKernel


def test_tp_shards_are_rejected(monkeypatch):
    monkeypatch.setattr(fp8_serving, "get_tensor_model_parallel_world_size", lambda: 2)
    method = object.__new__(fp8_serving.MasterFP8LinearMethod)
    with pytest.raises(ValueError, match="TP=1"):
        method.create_weights(torch.nn.Module(), 64, [32], 64, 32, torch.bfloat16)
    embedding_method = fp8_serving.MasterFP8EmbeddingMethod()
    with pytest.raises(ValueError, match="TP=1"):
        embedding_method.create_weights(torch.nn.Module(), 64, [32], 64, 32, torch.bfloat16)


def test_activation_contract_is_bf16(monkeypatch):
    monkeypatch.setattr(fp8_serving, "get_tensor_model_parallel_world_size", lambda: 1)
    method = object.__new__(fp8_serving.MasterFP8LinearMethod)
    with pytest.raises(ValueError, match="dtype bfloat16"):
        method.create_weights(torch.nn.Module(), 64, [32], 64, 32, torch.float16)


def test_shared_forward_is_scoped_to_owned_projection_kernel(monkeypatch):
    from rlforge import fp8

    monkeypatch.setenv("RLFORGE_FP8_FORWARD", "torch")
    monkeypatch.setattr(fp8_serving, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(OnlineLinearBase, "create_weights",
        lambda self, layer, *args, **kwargs: setattr(layer, "weight",
            torch.nn.Parameter(torch.empty(32, 64, dtype=torch.float32))))
    kernel = SimpleNamespace(logical_output_size=32)
    monkeypatch.setattr("vllm.model_executor.layers.quantization.online.fp8.init_fp8_linear_kernel",
                        lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(fp8_serving, "init_fp8_linear_kernel", lambda **kwargs: kernel)
    original = fp8_serving.CutlassFP8ScaledMMLinearKernel.apply_scaled_mm
    method = object.__new__(fp8_serving.MasterFP8LinearMethod)
    method.input_dtype = method.out_dtype = torch.bfloat16
    method.create_weights(torch.nn.Module(), 64, [32], 64, 32, torch.bfloat16)
    a = torch.zeros(7, 64, dtype=torch.float8_e4m3fn)
    weight = torch.zeros(32, 64, dtype=a.dtype)
    sa, sb = torch.ones(7, 1), torch.ones(1, 32)
    result = torch.randn(7, 32, dtype=torch.bfloat16)

    def forward(actual_a, actual_b, actual_sa, actual_sb):
        assert actual_a is a and actual_b.data_ptr() == weight.data_ptr()
        assert actual_b.stride() == weight.stride()
        assert actual_sa is sa and torch.equal(actual_sb, sb.t())
        return result

    monkeypatch.setattr(fp8, "forward_mm", forward)
    actual = kernel.apply_scaled_mm(A=a, B=weight.t(), As=sa, Bs=sb,
        out_dtype=torch.bfloat16, bias=None, output_shape=(1, 7, 32))
    assert torch.equal(actual, result.reshape(1, 7, 32))
    assert fp8_serving.CutlassFP8ScaledMMLinearKernel.apply_scaled_mm is original
    with pytest.raises(ValueError, match="without bias"):
        kernel.apply_scaled_mm(A=a, B=weight.t(), As=sa, Bs=sb,
            out_dtype=torch.bfloat16, bias=torch.zeros(32), output_shape=(7, 32))


def test_sensitive_projections_stay_unquantized_and_vocab_head_uses_master_method(monkeypatch):
    monkeypatch.setattr(fp8_serving, "LinearBase", torch.nn.Linear)
    config = fp8_serving.MasterFP8Config()
    layer = torch.nn.Linear(64, 32)
    layer.input_size, layer.output_size = 64, 32
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.linear_attn.in_proj_ba"),
        UnquantizedLinearMethod,
    )
    assert config.get_quant_method(torch.nn.Embedding(64, 32), "lm_head") is None
    assert isinstance(config.get_quant_method(layer, "lm_head"), UnquantizedLinearMethod)
    for cls, prefix in ((VocabParallelEmbedding, "model.embed_tokens"),
                        (ParallelLMHead, "lm_head")):
        vocab_layer = object.__new__(cls)
        torch.nn.Module.__init__(vocab_layer)
        assert isinstance(config.get_quant_method(vocab_layer, prefix),
                          fp8_serving.MasterFP8EmbeddingMethod)
    assert isinstance(config.get_quant_method(layer, "visual.blocks.0.proj"), UnquantizedLinearMethod)
    layer.output_size = 31
    assert isinstance(config.get_quant_method(layer, "model.layers.0.proj"), UnquantizedLinearMethod)


def test_tied_embedding_keeps_fp32_master_and_returns_bf16(monkeypatch):
    monkeypatch.setattr(fp8_serving, "get_tensor_model_parallel_world_size", lambda: 1)
    method = fp8_serving.MasterFP8EmbeddingMethod()
    embedding = _vocab_layer(VocabParallelEmbedding, method)
    head = _vocab_layer(ParallelLMHead, method)
    assert method.tie_weights(head, embedding) is head
    master = torch.randn(32, 64, generator=torch.Generator().manual_seed(11))
    embedding.weight.data.copy_(master)
    parameter = embedding.weight
    method.process_weights_after_loading(embedding)
    ids = torch.tensor([[0, 3, 31]])
    output = method.embedding(embedding, ids)
    assert head.weight is embedding.weight is parameter
    assert parameter.dtype == torch.float32
    assert torch.equal(parameter, master)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, master[ids].bfloat16())
    assert not hasattr(embedding, "_poor_rl_head_cache")


def test_tying_upgrades_qwen_embedding_created_without_quant_config(monkeypatch):
    from rlforge.fp8 import quantize_rows
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload, initialize_layerwise_reload, record_metadata_for_reloading,
    )
    from vllm.model_executor.models.utils import AutoWeightsLoader

    monkeypatch.setattr(fp8_serving, "get_tensor_model_parallel_world_size", lambda: 1)
    method = fp8_serving.MasterFP8EmbeddingMethod()
    model = torch.nn.Module()
    model.embed_tokens = _vocab_layer(VocabParallelEmbedding, UnquantizedEmbeddingMethod())
    model.lm_head = _vocab_layer(ParallelLMHead, method)
    assert model.embed_tokens.weight.dtype == torch.bfloat16
    parameter = model.embed_tokens.weight
    method.tie_weights(model.lm_head, model.embed_tokens)
    assert model.embed_tokens.weight is model.lm_head.weight is parameter
    assert parameter.dtype == model.embed_tokens.params_dtype == torch.float32
    assert isinstance(model.embed_tokens.quant_method, fp8_serving.MasterFP8EmbeddingMethod)
    generator = torch.Generator().manual_seed(31)
    initial = torch.randn(32, 64, generator=generator)
    record_metadata_for_reloading(model)
    parameter.data.copy_(initial)
    model.embed_tokens.quant_method.process_weights_after_loading(model.embed_tokens)
    cache = model.lm_head._poor_rl_head_cache
    pointers = parameter.data_ptr(), cache[0].data_ptr(), cache[1].data_ptr()
    updated = torch.randn(32, 64, generator=generator) * 7
    initialize_layerwise_reload(model)
    AutoWeightsLoader(model).load_weights([("embed_tokens.weight", updated)])
    finalize_layerwise_reload(model, None)
    quantized, scales = quantize_rows(updated)
    assert model.embed_tokens.weight is model.lm_head.weight is parameter
    assert torch.equal(parameter, updated)
    assert pointers == (parameter.data_ptr(), cache[0].data_ptr(), cache[1].data_ptr())
    assert torch.equal(cache[0].view(torch.uint8), quantized.view(torch.uint8))
    assert torch.equal(cache[1], scales)


def test_head_forward_uses_shared_quantization_and_bf16_output(monkeypatch):
    from rlforge import fp8

    monkeypatch.setattr(fp8_serving, "get_tensor_model_parallel_world_size", lambda: 1)
    method = fp8_serving.MasterFP8EmbeddingMethod()
    head = _vocab_layer(ParallelLMHead, method)
    generator = torch.Generator().manual_seed(13)
    master = torch.randn(32, 64, generator=generator)
    head.weight.data.copy_(master)
    method.process_weights_after_loading(head)
    expected_scale = _fp8_channel_scale(master.abs().amax(-1, keepdim=True))
    expected_weight = _fp8_quant_per_channel(master, expected_scale)
    weight, scale = head._poor_rl_head_cache
    assert torch.equal(weight.view(torch.uint8), expected_weight.view(torch.uint8))
    assert torch.equal(scale, expected_scale)
    inputs = torch.randn(8, 64, generator=generator, dtype=torch.bfloat16)
    input_scale = _fp8_channel_scale(inputs.float().abs().amax(-1, keepdim=True))
    input_quantized = _fp8_quant_per_channel(inputs, input_scale)
    recorded = {}

    def forward_mm(a, b, scale_a, scale_b):
        recorded.update(a=a, b=b, scale_a=scale_a, scale_b=scale_b)
        return ((a.float() * scale_a) @ (b.float() * scale_b).t()).bfloat16()

    monkeypatch.setattr(fp8, "forward_mm", forward_mm)
    bias = torch.randn(32, generator=generator)
    output = method.apply(head, inputs, bias)
    assert torch.equal(recorded["a"].view(torch.uint8), input_quantized.view(torch.uint8))
    assert torch.equal(recorded["scale_a"], input_scale)
    assert recorded["b"] is weight and recorded["scale_b"] is scale
    expected = ((input_quantized.float() * input_scale)
                @ (expected_weight.float() * expected_scale).t()).bfloat16()
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, (expected.float() + bias).bfloat16())


def test_head_hot_reload_refreshes_cache_and_preserves_storage(monkeypatch):
    from rlforge.fp8 import quantize_rows
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload, initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    monkeypatch.setattr(fp8_serving, "get_tensor_model_parallel_world_size", lambda: 1)
    method = fp8_serving.MasterFP8EmbeddingMethod()
    head = _vocab_layer(ParallelLMHead, method)
    generator = torch.Generator().manual_seed(17)
    head.weight.data.copy_(torch.randn(32, 64, generator=generator))
    record_metadata_for_reloading(head)
    method.process_weights_after_loading(head)
    master = head.weight
    cache = head._poor_rl_head_cache
    pointers = master.data_ptr(), cache[0].data_ptr(), cache[1].data_ptr()
    initial_quantized = cache[0].clone()
    initial_scale = cache[1].clone()
    updated = torch.randn(32, 64, generator=generator) * 7
    initialize_layerwise_reload(head)
    assert head.weight.dtype == torch.float32 and head.weight.is_meta
    head.weight.weight_loader(head.weight, updated)
    finalize_layerwise_reload(head, None)
    expected_weight, expected_scale = quantize_rows(updated)
    assert head.weight is master and head._poor_rl_head_cache is cache
    assert pointers == (master.data_ptr(), cache[0].data_ptr(), cache[1].data_ptr())
    assert torch.equal(master, updated)
    assert torch.equal(cache[0].view(torch.uint8), expected_weight.view(torch.uint8))
    assert torch.equal(cache[1], expected_scale)
    assert not torch.equal(initial_quantized.view(torch.uint8), cache[0].view(torch.uint8))
    assert not torch.equal(initial_scale, cache[1])


def test_tied_head_reload_refreshes_cache_when_only_embedding_weight_is_loaded(monkeypatch):
    from rlforge.fp8 import quantize_rows
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload, initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm.model_executor.models.utils import AutoWeightsLoader

    monkeypatch.setattr(fp8_serving, "get_tensor_model_parallel_world_size", lambda: 1)
    method = fp8_serving.MasterFP8EmbeddingMethod()
    model = torch.nn.Module()
    model.embed_tokens = _vocab_layer(VocabParallelEmbedding, method)
    model.lm_head = _vocab_layer(ParallelLMHead, method)
    method.tie_weights(model.lm_head, model.embed_tokens)
    generator = torch.Generator().manual_seed(23)
    model.embed_tokens.weight.data.copy_(torch.randn(32, 64, generator=generator))
    record_metadata_for_reloading(model)
    method.process_weights_after_loading(model.embed_tokens)
    method.process_weights_after_loading(model.lm_head)
    master = model.embed_tokens.weight
    cache = model.lm_head._poor_rl_head_cache
    pointers = master.data_ptr(), cache[0].data_ptr(), cache[1].data_ptr()
    for _ in range(2):
        updated = torch.randn(32, 64, generator=generator) * 7
        initialize_layerwise_reload(model)
        loaded = AutoWeightsLoader(model).load_weights([("embed_tokens.weight", updated)])
        finalize_layerwise_reload(model, None)
        expected_weight, expected_scale = quantize_rows(updated)
        assert loaded == {"embed_tokens.weight"}
        assert model.embed_tokens.weight is model.lm_head.weight is master
        assert model.lm_head._poor_rl_head_cache is cache
        assert pointers == (master.data_ptr(), cache[0].data_ptr(), cache[1].data_ptr())
        assert torch.equal(master, updated)
        assert torch.equal(cache[0].view(torch.uint8), expected_weight.view(torch.uint8))
        assert torch.equal(cache[1], expected_scale)


def test_hot_reload_refreshes_quantized_weights_and_scales_in_place():
    from vllm.model_executor.model_loader.reload.layerwise import (
        initialize_layerwise_reload, record_metadata_for_reloading,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(torch.randn(32, 64), requires_grad=False),
    )
    layer.weight.weight_loader = default_weight_loader
    layer.tp_size = 1
    layer.input_size = layer.input_size_per_partition = 64
    layer.output_size = layer.output_size_per_partition = 32
    method = object.__new__(fp8_serving.MasterFP8LinearMethod)
    method.fp8_linear = SimpleNamespace(process_weights_after_loading=lambda layer: None)
    layer.quant_method = method
    record_metadata_for_reloading(layer)
    method.process_weights_after_loading(layer)
    weight_ptr, scale_ptr = layer.weight.data_ptr(), layer.weight_scale.data_ptr()
    initial_scale = layer.weight_scale.clone()

    initialize_layerwise_reload(layer)
    assert layer.weight.dtype == torch.float32 and layer.weight.is_meta
    updated = torch.randn(32, 64) * 7
    layer.weight.weight_loader(layer.weight, updated)
    expected_scale = _fp8_channel_scale(updated.abs().amax(-1, keepdim=True))
    expected_q = _fp8_quant_per_channel(updated, expected_scale).t()
    assert layer.weight.data_ptr() == weight_ptr
    assert layer.weight_scale.data_ptr() == scale_ptr
    assert torch.equal(layer.weight_scale, expected_scale)
    assert torch.equal(layer.weight.view(torch.uint8), expected_q.view(torch.uint8))
    assert not torch.equal(initial_scale, layer.weight_scale)


def test_rope_rounding_requires_an_explicit_fp8_config_opt_in(monkeypatch):
    from rlforge import serving_rope

    calls = []
    monkeypatch.setattr(serving_rope, "install", lambda: calls.append(True))
    monkeypatch.delenv("RLFORGE_FP8_ROPE_BF16", raising=False)
    fp8_serving.MasterFP8Config()
    assert calls == []
    monkeypatch.setenv("RLFORGE_FP8_ROPE_BF16", "1")
    fp8_serving.MasterFP8Config.from_config({})
    assert calls == [True]


def test_shared_rope_preserves_default_and_fp16_routes(monkeypatch, tmp_path):
    import hashlib
    from rlforge import serving_rope
    from vllm.model_executor.layers import fused_qk_norm_rope as native_module
    from vllm.model_executor.models import qwen3_next

    received = []

    def native(*args, **kwargs):
        received.append("native")
        return ("native",)

    def rounded(*args, **kwargs):
        received.append("shared")
        return ("shared",)

    source = tmp_path / "native_rope.py"
    source.write_text("source-pinned RoPE fixture")
    monkeypatch.setattr(native_module, "__file__", str(source))
    monkeypatch.setattr(native_module, "fused_qk_rmsnorm_rope_gate", native)
    monkeypatch.setattr(qwen3_next, "fused_qk_rmsnorm_rope_gate", native)
    monkeypatch.setattr(serving_rope, "_INSTALLED", None)
    monkeypatch.setattr(serving_rope, "NATIVE_SOURCE_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(serving_rope, "fused_qk_rmsnorm_rope_gate", rounded)
    monkeypatch.delenv("RLFORGE_FP8_ROPE_BF16", raising=False)
    assert serving_rope.install() is False
    assert qwen3_next.fused_qk_rmsnorm_rope_gate is native

    monkeypatch.setenv("RLFORGE_FP8_ROPE_BF16", "1")
    fp8_serving.MasterFP8Config()
    shared = qwen3_next.fused_qk_rmsnorm_rope_gate
    assert shared is not native
    assert serving_rope.install() is True
    assert qwen3_next.fused_qk_rmsnorm_rope_gate is shared
    assert shared(torch.zeros(1, dtype=torch.bfloat16)) == ("shared",)
    assert shared(q_gate=torch.zeros(1, dtype=torch.bfloat16)) == ("shared",)
    assert shared(torch.zeros(1, dtype=torch.float16)) == ("native",)
    assert received == ["shared", "shared", "native"]
    monkeypatch.setattr(qwen3_next, "fused_qk_rmsnorm_rope_gate", native)
    with pytest.raises(RuntimeError, match="replaced after installation"):
        serving_rope.install()


def test_shared_rope_refuses_changed_source_or_replaced_native(monkeypatch, tmp_path):
    import hashlib
    from rlforge import serving_rope
    from vllm.model_executor.layers import fused_qk_norm_rope as native_module
    from vllm.model_executor.models import qwen3_next

    source = tmp_path / "native_rope.py"
    source.write_text("changed native RoPE")
    native = lambda *args, **kwargs: None
    monkeypatch.setattr(native_module, "__file__", str(source))
    monkeypatch.setattr(native_module, "fused_qk_rmsnorm_rope_gate", native)
    monkeypatch.setattr(qwen3_next, "fused_qk_rmsnorm_rope_gate", native)
    monkeypatch.setattr(serving_rope, "_INSTALLED", None)
    monkeypatch.setattr(serving_rope, "NATIVE_SOURCE_SHA256", "changed")
    monkeypatch.setenv("RLFORGE_FP8_ROPE_BF16", "1")
    with pytest.raises(RuntimeError, match="source changed"):
        serving_rope.install()
    assert qwen3_next.fused_qk_rmsnorm_rope_gate is native
    monkeypatch.setattr(serving_rope, "NATIVE_SOURCE_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest())
    replacement = lambda *args, **kwargs: None
    monkeypatch.setattr(qwen3_next, "fused_qk_rmsnorm_rope_gate", replacement)
    with pytest.raises(RuntimeError, match="already replaced"):
        serving_rope.install()
    assert qwen3_next.fused_qk_rmsnorm_rope_gate is replacement
