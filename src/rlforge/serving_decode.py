"""Source-pinned ordinary vLLM decode scheduling and sampled-output bookkeeping.

Opt in with RLFORGE_SERVING_DECODE=1. Unsupported scheduler/request/cache
boundaries use untouched native methods. Derived from frozen 3c6a1c bulk and
6077b3 per-request prototypes; fallback calls directly native vLLM in this
consolidation. Prior experiment throughput does not validate this new recipe.
"""
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import textwrap

INSTALLED = False
ENABLED = False
COUNTS = dict(fast_steps=0, native_steps=0, no_op_allocations=0, boundary_allocations=0,
              output_scopes=0, no_op_caches=0, native_caches=0,
              fast_requests=0, native_requests=0)
OUTPUT_SOURCE_SHA = 'ac3f8f3f3a48ad50089e0dff3fd0f56bebf262e1664a788e23b86fdf3d4cd0d6'


def register():
    if os.environ.get('RLFORGE_SERVING_DECODE') != '1':
        return False
    return install()


def _output_function(native):
    source = inspect.getsource(native)
    if hashlib.sha256(source.encode()).hexdigest() != OUTPUT_SOURCE_SHA:
        raise RuntimeError('Native vLLM output source changed; decode optimization refused')
    anchor = '            scheduled_spec_token_ids = (\n'
    if source.count(anchor) != 1:
        raise RuntimeError('Native vLLM output loop anchor changed')
    branch = """
            params=request.sampling_params
            if (_decode_enabled() and num_tokens_scheduled==1
                    and not output_is_stale and request.status==RequestStatus.RUNNING
                    and len(generated_token_ids)==1 and params is not None
                    and not request.resumable and not request.mm_features
                    and request.structured_output_request is None and not request.pooling_params
                    and params.repetition_detection is None and request.prefill_stats is None
                    and not request.is_prefill_chunk and not self.is_mm_encoder_only
                    and not self.enable_return_routed_experts and not self.return_sampling_mask
                    and not scheduler_output.scheduled_spec_decode_tokens
                    and not self.num_spec_tokens and self.num_sampled_tokens_per_step==1
                    and pooler_outputs is None and routing_data is None
                    and req_id not in prompt_logprobs_dict
                    and request.num_output_placeholders>=1
                    and request.num_computed_tokens>=request.num_prompt_tokens
                    and len(request._all_token_ids)+1<self.max_model_len
                    and len(request._output_token_ids)+1<request.max_tokens
                    and generated_token_ids[0]!=params.eos_token_id
                    and generated_token_ids[0] not in (params.stop_token_ids or ())):
                token=generated_token_ids[0]
                request._output_token_ids.append(token)
                request._all_token_ids.append(token)
                if request._block_hasher is not None:
                    request.block_hashes.extend(request._block_hasher(request))
                request.num_output_placeholders-=1
                self.kv_cache_manager.cache_blocks(request,
                    request.num_computed_tokens-request.num_output_placeholders)
                fast_logprobs=None
                if params.num_logprobs is not None and logprobs:
                    fast_logprobs=logprobs.slice_request(req_index,1)
                if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                    request.num_nans_in_logits=num_nans_in_logits[req_id]
                fast_events=request.events
                if fast_events:
                    request.events=[]
                else:
                    fast_events=None
                outputs[request.client_index].append(EngineCoreOutput(
                    request_id=req_id,new_token_ids=generated_token_ids,new_logprobs=fast_logprobs,
                    stop_reason=request.stop_reason,events=fast_events,trace_headers=request.trace_headers,
                    num_nans_in_logits=request.num_nans_in_logits))
                COUNTS['fast_requests']+=1
                continue
            COUNTS['native_requests']+=1
"""
    namespace = dict(native.__globals__)
    namespace.update(_decode_enabled=lambda: ENABLED, COUNTS=COUNTS)
    source = source.replace(anchor, branch + anchor)
    exec(compile(textwrap.dedent(source), 'rlforge_native_decode_output', 'exec'), namespace)
    return namespace['update_from_output']


def install():
    global INSTALLED, ENABLED
    if INSTALLED:
        return True
    if os.environ.get('RLFORGE_SERVING_DECODE') != '1':
        return False
    from vllm.v1.core.sched.scheduler import Scheduler, PauseState
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    from vllm.v1.core.block_pool import make_block_hash_with_group_id
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.request import Request, RequestStatus
    native_sources = {
        'vllm.v1.request': 'da610d5203e72d2d524adc5fb39df11502a23520f2f31d9a44cf13d343f9c25d',
        'vllm.v1.core.sched.scheduler': 'c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884',
        'vllm.v1.core.sched.async_scheduler': 'e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508',
        'vllm.v1.core.kv_cache_coordinator': 'bf342f79f2bc8d6acb69d3becefb334509e84cae888d42bbd421a8dc8ca9eca3',
        'vllm.v1.core.single_type_kv_cache_manager': '0233531d85aeced9e57a38c76d83db970519c92902b7e55484390cb2b64985bf',
        'vllm.v1.core.kv_cache_manager': '58800535eda854580d84e1025bf32fe01b2a43ce0bba501b4f3d125e4f2c8afa',
    }
    for name, wanted in native_sources.items():
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != wanted:
            raise RuntimeError('Native source changed: ' + name)
    native_schedule = Scheduler.schedule
    original_output = Scheduler.update_from_output
    scheduler_file = importlib.import_module('vllm.v1.core.sched.scheduler').__file__
    if native_schedule.__code__.co_filename != scheduler_file:
        raise RuntimeError('Native vLLM schedule already replaced; decode optimization refused')
    native_output = _output_function(original_output)
    def layout(manager):
        coordinator = manager.coordinator
        if (type(coordinator) is not HybridKVCacheCoordinator or not manager.enable_caching
                or coordinator.eagle_group_ids):
            return None
        size = coordinator.scheduler_block_size
        attention, recurrent = [], []
        for item in coordinator.single_type_managers:
            if (item.block_size != size or size % item.block_pool.hash_block_size
                    or not item.enable_caching or item.use_eagle or item._partial_hit_reqs):
                return None
            if type(item) is FullAttentionManager and type(item.kv_cache_spec) is FullAttentionSpec:
                attention.append(item)
            elif (type(item) is MambaManager and item.mamba_cache_mode == 'align'
                  and not item.num_speculative_blocks and not item.has_prefill_checkpoint_blocks
                  and not item._checkpoints):
                recurrent.append(item)
            else:
                return None
        return (size, attention, recurrent) if attention and recurrent else None

    def partial_cache_noop(request, tokens, attention, recurrent, size):
        for item in attention:
            hash_size = item.block_pool.hash_block_size
            boundary = request.num_prompt_tokens // hash_size * hash_size
            if hash_size == size or not boundary or boundary > tokens or not boundary % size:
                continue
            table = item.req_to_blocks.get(request.request_id, ())
            index = boundary // size
            if index >= len(table) or table[index].is_null:
                continue
            hash_index = boundary // hash_size - 1
            if hash_index >= len(request.block_hashes):
                return False
            key = make_block_hash_with_group_id(request.block_hashes[hash_index], item.kv_cache_group_id)
            block = table[index]
            if block.block_hash != key and not item.block_pool.cached_block_hash_to_block.contain(key, block.block_id):
                return False
        for item in recurrent:
            hash_size = item.block_pool.hash_block_size
            boundary = request.num_prompt_tokens // hash_size * hash_size
            if tokens == boundary and tokens % size:
                return False
        return True

    def schedule(self, throttle_prefills=False):
        running = self.running
        manager = self.kv_cache_manager
        config = self.scheduler_config
        supported = (ENABLED and type(self) is AsyncScheduler and running and not self.waiting
            and not self.skipped_waiting and self._pause_state == PauseState.UNPAUSED
            and self.use_v2_model_runner and not self.use_pp
            and not self.vllm_config.speculative_config and not self.num_spec_tokens
            and not self.num_lookahead_tokens and not self.num_prefill_lookahead
            and self.num_sampled_tokens_per_step == 1 and self.connector is None
            and self.ec_connector is None and self.lora_config is None
            and not self.is_encoder_decoder and not self.enable_return_routed_experts
            and not self.return_sampling_mask and not self.dynamic_sd_lookup
            and not self._inflight_prefills and not self.num_waiting_for_streaming_input
            and len(running) <= self.max_num_running_reqs
            and len(running) <= min(self.max_num_scheduled_tokens, config.max_num_batched_tokens)
            and not (self.log_stats and self.observability_config.enable_logging_iteration_details))
        context = layout(manager) if supported else None
        if context is None:
            COUNTS['native_steps'] += 1
            return native_schedule(self, throttle_prefills=throttle_prefills)
        size, attention, recurrent = context
        fine_hash = any(item.block_pool.hash_block_size != size for item in attention + recurrent)
        partial = manager.coordinator.enable_partial_hash_hits
        if manager.block_pool.get_num_free_blocks() < 2 * len(running) * (len(attention) + len(recurrent)) + manager.watermark_blocks:
            COUNTS['native_steps'] += 1
            return native_schedule(self, throttle_prefills=throttle_prefills)
        req_ids, new_blocks, computed_tokens, output_tokens, boundary_requests = [], [], [], [], []
        scheduled = {}
        step = self.current_step + 1
        for request in running:
            if type(request) is not Request:
                COUNTS['native_steps'] += 1
                return native_schedule(self, throttle_prefills=throttle_prefills)
            t = request.num_computed_tokens
            placeholders = request.num_output_placeholders
            total = len(request._all_token_ids)
            n_output = len(request._output_token_ids)
            if (request.status != RequestStatus.RUNNING or request.resumable
                    or request.is_prefill_chunk or request.mm_features
                    or request.structured_output_request is not None or request.spec_token_ids
                    or request.next_decode_eligible_step > step or t < request.num_prompt_tokens
                    or t >= self.max_model_len - 1 or total + placeholders - t != 1
                    or placeholders > 0 and t + 2 - placeholders >= request.num_prompt_tokens + request.max_tokens
                    or t < total - 1):
                COUNTS['native_steps'] += 1
                return native_schedule(self, throttle_prefills=throttle_prefills)
            rid = request.request_id
            required = (t + size) // size
            cache_tokens = min(t + 1, total)
            cache_tokens = cache_tokens if partial else cache_tokens // size * size
            cached_required = cache_tokens // size
            processed = max(0, t - request.num_in_flight_tokens)
            retired_required = max(0, processed - 1) // size
            old_state_limit = (processed + size - 1) // size - 1
            noop = True
            for item in attention:
                table = item.req_to_blocks.get(rid)
                cached = item.num_cached_block.get(rid)
                if not table or cached is None or required > len(table) or cached < cached_required:
                    noop = False
                    break
            if noop:
                for item in recurrent:
                    table = item.req_to_blocks.get(rid)
                    cached = item.num_cached_block.get(rid)
                    if (not table or cached is None or required != len(table) or cached < cached_required
                            or rid not in item._allocated_block_reqs
                            or min(retired_required, len(table)) > item._num_retired_blocks.get(rid, 0)):
                        noop = False
                        break
                    old_state = item.last_state_block_idx.get(rid)
                    if old_state is not None and old_state < old_state_limit and not table[old_state].is_null:
                        noop = False
                        break
            if noop and fine_hash and not partial_cache_noop(request, cache_tokens, attention, recurrent, size):
                noop = False
            if not noop:
                boundary_requests.append((len(req_ids), request))
            req_ids.append(rid)
            new_blocks.append(None)
            scheduled[rid] = 1
            computed_tokens.append(t)
            output_tokens.append(n_output + placeholders)
        self.current_step += 1
        self.prefill_capacity_bound = False
        manager.new_step_starts()
        for index, request in boundary_requests:
            result = manager.allocate_slots(request, 1, num_lookahead_tokens=0)
            if result is None:
                raise RuntimeError('Bulk decode free-block proof failed')
            new_blocks[index] = result.get_block_ids(allow_none=True)
        manager.take_boundary_state_offloads()
        copies, retained = manager.take_kv_cache_block_copies()
        if copies:
            self._free_cow_retained_blocks(retained, self.sched_step_seq + 1)
        result = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(req_ids=req_ids, resumed_req_ids=set(),
                new_token_ids=[], all_token_ids={}, new_block_ids=new_blocks,
                num_computed_tokens=computed_tokens, num_output_tokens=output_tokens),
            num_scheduled_tokens=scheduled, total_num_scheduled_tokens=len(running),
            scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
            num_common_prefix_blocks=manager.get_num_common_prefix_blocks(running[0].request_id),
            preempted_req_ids=self.reset_preempted_req_ids, finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(), kv_cache_block_copies=copies or None,
            num_spec_tokens_to_schedule=0, ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata())
        if self.defer_block_free:
            self.sched_step_seq += 1
        self._update_after_schedule(result)
        COUNTS['fast_steps'] += 1
        COUNTS['no_op_allocations'] += len(running) - len(boundary_requests)
        COUNTS['boundary_allocations'] += len(boundary_requests)
        return result

    def output_step(self, scheduler_output, model_output):
        coordinator = self.kv_cache_manager.coordinator
        context = layout(self.kv_cache_manager) if ENABLED else None
        if context is None:
            return native_output(self, scheduler_output, model_output)
        size, attention, recurrent = context
        managers = attention + recurrent
        fine_hash = any(item.block_pool.hash_block_size != size for item in managers)
        partial = coordinator.enable_partial_hash_hits
        old_instance_method = coordinator.__dict__.get('cache_blocks')
        native_cache = coordinator.cache_blocks

        def cache(request, tokens):
            aligned = tokens if partial else tokens // size * size
            needed = aligned // size
            if (type(request) is Request and request.num_computed_tokens >= request.num_prompt_tokens
                    and all(item.num_cached_block.get(request.request_id, 0) >= needed for item in managers)
                    and (not fine_hash or partial_cache_noop(request, aligned, attention, recurrent, size))):
                COUNTS['no_op_caches'] += 1
                return
            COUNTS['native_caches'] += 1
            return native_cache(request, tokens)

        coordinator.cache_blocks = cache
        try:
            COUNTS['output_scopes'] += 1
            return native_output(self, scheduler_output, model_output)
        finally:
            if old_instance_method is None:
                del coordinator.cache_blocks
            else:
                coordinator.cache_blocks = old_instance_method

    Scheduler.schedule = schedule
    Scheduler.update_from_output = output_step
    INSTALLED = True
    ENABLED = True
    return True
