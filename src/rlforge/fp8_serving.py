"""vLLM FP8 serving from FP32 master weights, with rowwise W8A8 quantization.

Registered through the ``vllm.general_plugins`` entry point. Select
``--quantization poor_rl_fp8`` with TP=1 and BF16 KV/recurrent state.
Set ``RLFORGE_FP8_ROPE_BF16=1`` to share BF16 RoPE product rounding.
"""

import os
from types import MethodType

import torch
from weakref import WeakSet

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm import CutlassFP8ScaledMMLinearKernel
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, UnquantizedEmbeddingMethod, VocabParallelEmbedding,
)


class MasterFP8EmbeddingMethod(UnquantizedEmbeddingMethod):
    """Keep tied embedding/head masters FP32; quantize only the head GEMM."""
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("poor_rl_fp8 requires TP=1")
        super().create_weights(layer, input_size_per_partition, output_partition_sizes,
                               input_size, output_size, torch.float32, **attrs)

    def tie_weights(self, layer, embed_tokens):
        # Qwen3.5 constructs its embedding without quant_config, before checkpoint loading.
        if not isinstance(embed_tokens.quant_method, MasterFP8EmbeddingMethod):
            embed_tokens.weight.data = embed_tokens.weight.data.to(torch.float32)
            embed_tokens.params_dtype = torch.float32
            embed_tokens.quant_method = MasterFP8EmbeddingMethod()
        super().tie_weights(layer, embed_tokens)
        if not hasattr(embed_tokens, "_poor_rl_tied_heads"):
            embed_tokens._poor_rl_tied_heads = WeakSet()
        embed_tokens._poor_rl_tied_heads.add(layer)
        return layer

    def process_weights_after_loading(self, layer):
        from rlforge.fp8 import quantize_rows
        heads = (layer,) if isinstance(layer, ParallelLMHead) else tuple(
            getattr(layer, "_poor_rl_tied_heads", ()))
        if not heads:
            return
        # A tied head has no separate checkpoint tensor and may skip its reload hook.
        quantized, scales = quantize_rows(layer.weight.detach())
        for head in heads:
            cache = getattr(head, "_poor_rl_head_cache", None)
            if cache is None:
                head._poor_rl_head_cache = quantized.clone(), scales.clone()
            else:
                cache[0].copy_(quantized)
                cache[1].copy_(scales)

    def embedding(self, layer, inputs):
        return torch.nn.functional.embedding(inputs, layer.weight).to(torch.bfloat16)

    def apply(self, layer, x, bias=None):
        from rlforge.fp8 import forward_mm, quantize_rows
        quantized, scales = quantize_rows(x.reshape(-1, x.shape[-1]).to(torch.bfloat16))
        weight, weight_scale = layer._poor_rl_head_cache
        result = forward_mm(quantized, weight, scales, weight_scale)
        if bias is not None:
            result = (result.float() + bias.float()).to(torch.bfloat16)
        return result


class MasterFP8LinearMethod(Fp8PtpcOnlineLinearMethod):
    def create_weights(
        self, layer, input_size_per_partition, output_partition_sizes,
        input_size, output_size, params_dtype, **extra_weight_attrs,
    ):
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError(
                "poor_rl_fp8 requires TP=1: row-parallel activation scales "
                "otherwise depend on the serving shard. Use data parallel rollout."
            )
        if params_dtype != torch.bfloat16:
            raise ValueError(
                "poor_rl_fp8 requires --dtype bfloat16 for the shared activation contract."
            )
        backend = os.environ.get("RLFORGE_FP8_FORWARD", "native")
        if backend not in ("native", "torch"):
            raise ValueError("RLFORGE_FP8_FORWARD must be native or torch")
        # Layerwise hot reload restores this metadata, so every new FP32 policy
        # reaches the quantizer without an intermediate BF16 weight round.
        super().create_weights(
            layer, input_size_per_partition, output_partition_sizes,
            input_size, output_size, torch.float32, **extra_weight_attrs,
        )
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            force_kernel=CutlassFP8ScaledMMLinearKernel,
            module_name=type(self).__name__,
        )
        if backend == "torch":
            self.fp8_linear.apply_scaled_mm = MethodType(_shared_scaled_mm, self.fp8_linear)


def _shared_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):
    from rlforge.fp8 import forward_mm

    if (out_dtype != torch.bfloat16 or bias is not None
            or B.shape != (A.shape[1], self.logical_output_size)):
        raise ValueError("Shared FP8 forward requires aligned BF16 output without bias")
    scales = Bs.t() if Bs.ndim == 2 else Bs
    result = forward_mm(A, B.t(), As, scales)
    return result.view(*output_shape[:-1], self.logical_output_size)


class MasterFP8Config(QuantizationConfig):
    def __init__(self):
        if os.environ.get("RLFORGE_FP8_ROPE_BF16") == "1":
            from rlforge.serving_rope import install
            install()

    @classmethod
    def get_name(cls):
        return "poor_rl_fp8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 90

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, VocabParallelEmbedding):
            return MasterFP8EmbeddingMethod()
        if isinstance(layer, LinearBase):
            if (prefix.rsplit(".", 1)[-1] in {"in_proj_ba", "lm_head"}
                    or {"vision", "visual", "vision_model"}.intersection(prefix.split("."))
                    or layer.input_size % 16 or layer.output_size % 16):
                return UnquantizedLinearMethod()
            return MasterFP8LinearMethod()
        if isinstance(layer, RoutedExperts):
            raise ValueError(
                "poor_rl_fp8 currently supports dense decoder projections; "
                "FP8 MoE expert training and serving need a separate numerical gate."
            )
        return None


def register():
    register_quantization_config(MasterFP8Config.get_name())(MasterFP8Config)
    from rlforge.serving_decode import register as register_decode
    register_decode()
