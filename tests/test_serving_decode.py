"""Compare optional decode bookkeeping directly with untouched native vLLM."""
from collections import defaultdict, deque
import hashlib
import importlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('vllm', reason='Native decode gates require the serving environment')
import numpy as np
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler, PauseState, SchedulingPolicy
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput, LogprobsLists
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from rlforge import serving_decode

def _base_fixture(prompt, count, tokens, pools=512):
    groups = [KVCacheGroupSpec(['attn'], FullAttentionSpec(544, num_kv_heads=1, head_size=8, dtype=torch.bfloat16))]
    for i in range(3):
        groups.append(KVCacheGroupSpec([str(i)], MambaSpec(544, ((2, 2),), (torch.float32,), mamba_cache_mode='align')))
    cache_config = KVCacheConfig(pools, [], groups)
    manager = KVCacheManager(cache_config, 8192, 544, 544)
    scheduler = AsyncScheduler.__new__(AsyncScheduler)
    scheduler.__dict__.update(
        running=[], waiting=create_request_queue(SchedulingPolicy.FCFS),
        skipped_waiting=create_request_queue(SchedulingPolicy.FCFS),
        requests={}, policy=SchedulingPolicy.FCFS, kv_cache_manager=manager,
        kv_cache_config=cache_config, scheduler_config=SimpleNamespace(
            max_num_batched_tokens=32768, long_prefill_token_threshold=0,
            async_scheduling=True),
        vllm_config=SimpleNamespace(speculative_config=None),
        cache_config=SimpleNamespace(block_size=544),
        observability_config=SimpleNamespace(enable_logging_iteration_details=False),
        _pause_state=PauseState.UNPAUSED, use_v2_model_runner=True, use_pp=False, pp_size=1,
        num_spec_tokens=0, num_lookahead_tokens=0, num_prefill_lookahead=0,
        connector=None, ec_connector=None, lora_config=None, is_encoder_decoder=False,
        enable_return_routed_experts=False, return_sampling_mask=False, dynamic_sd_lookup=None,
        _inflight_prefills=set(), num_waiting_for_streaming_input=0, max_num_running_reqs=2048,
        max_num_scheduled_tokens=32768, max_model_len=8192, num_sampled_tokens_per_step=1,
        log_stats=False, current_step=0, max_num_encoder_input_tokens=0,
        num_output_placeholders=0, prefill_capacity_bound=False,
        need_mamba_block_aligned_split=True, reset_preempted_req_ids=set(), finished_req_ids=set(),
        defer_block_free=False, needs_kv_cache_zeroing=False, _skip_zero_block_ids=set(),
        sched_step_seq=0, prev_step_scheduled_req_ids=set(),
        mamba_partial_cache_hit=False, mamba_has_prefill_checkpoint_blocks=False,
        use_eagle_block_drop=False, mamba_fine_grained_prefix_cache=False,
        mamba_prefill_checkpoint_alignment=None, hash_block_size=544,
        encoder_cache_manager=SimpleNamespace(get_freed_mm_hashes=lambda: [], get_manager_metadata=lambda: None))
    for i in range(count):
        def hasher(request):
            return [hashlib.sha256(bytes(request._all_token_ids[:end])).digest()
                    for end in range((len(request.block_hashes) + 1) * 544, request.num_tokens + 1, 544)]
        request = Request('r' + str(i), [i % 251] * prompt, SamplingParams(max_tokens=tokens),
                          None, block_hasher=hasher)
        request.status = RequestStatus.RUNNING
        manager.new_step_starts()
        assert manager.allocate_slots(request, prompt) is not None
        request.num_computed_tokens = prompt
        manager.cache_blocks(request, prompt)
        request.append_output_token_ids(7)
        scheduler.running.append(request)
        scheduler.requests[request.request_id] = request
    return scheduler
def _basic_state(scheduler):
    manager = scheduler.kv_cache_manager
    return {
        'current_step': scheduler.current_step,
        'prefill_capacity_bound': scheduler.prefill_capacity_bound,
        'requests': [(r.request_id, list(r.all_token_ids), r.num_computed_tokens,
                      r.num_in_flight_tokens, r.num_output_placeholders, r.is_prefill_chunk,
                      r.next_decode_eligible_step, r.status.name) for r in scheduler.running],
        'blocks': [(b.block_id, b.ref_cnt, b.is_null, str(b.block_hash), b.block_hash_num_tokens)
                   for b in manager.block_pool.blocks],
        'free': manager.block_pool.get_num_free_blocks(),
        'hash_lookup': sorted((str(k), sorted(v) if isinstance(v, dict) else [v.block_id])
                             for k, v in manager.block_pool.cached_block_hash_to_block._cache.items()),
        'managers': [
            {'req_blocks': {k: [b.block_id for b in v] for k, v in m.req_to_blocks.items()},
             'cached': dict(m.num_cached_block), 'retired': dict(getattr(m, '_num_retired_blocks', {})),
             'last': dict(getattr(m, 'last_state_block_idx', {})),
             'allocated': sorted(getattr(m, '_allocated_block_reqs', [])),
             'checkpoints': dict(getattr(m, '_checkpoints', {}))}
            for m in manager.coordinator.single_type_managers]}

def _fixture(prompt,count,events=False):
    scheduler=_base_fixture(prompt,count,4096,pools=24576)
    scheduler.__dict__.update(perf_metrics=None,is_mm_encoder_only=False,grammar_compile_error_reqs=set(),
        finished_req_ids_dict=defaultdict(set),kv_event_publisher=SimpleNamespace(publish=lambda batch:None),
        structured_output_manager=StructuredOutputManager.__new__(StructuredOutputManager))
    for i,request in enumerate(scheduler.running):
        request.prefill_stats=None
        request.client_index=i%2
        request.sampling_params.logprobs=0
    scheduler.encoder_cache_manager.free=lambda request:None
    return scheduler
def _model_output(scheduler,tick):
    ids=[r.request_id for r in scheduler.running]
    token=np.full((len(ids),1),(8+tick)%251,dtype=np.int32)
    return ModelRunnerOutput(req_ids=ids,req_id_to_index={rid:i for i,rid in enumerate(ids)},
        sampled_token_ids=token.tolist(),
        logprobs=LogprobsLists(token.copy(),np.arange(len(ids),dtype=np.float32).reshape(-1,1)/-71,
            np.ones(len(ids),dtype=np.int32)),num_nans_in_logits={ids[0]:tick%3} if tick%7==0 else None)
def _compare_output(left,right):
    assert left.keys()==right.keys()
    for client in left:
        a,b=left[client],right[client]
        assert a.finished_requests==b.finished_requests and a.scheduler_stats==b.scheduler_stats
        assert len(a.outputs)==len(b.outputs)
        for x,y in zip(a.outputs,b.outputs):
            for field in x.__struct_fields__:
                if field=='new_logprobs':
                    if x.new_logprobs is None or y.new_logprobs is None:
                        assert x.new_logprobs is y.new_logprobs
                    else:
                        for u,v in zip(x.new_logprobs,y.new_logprobs):
                            if isinstance(u,np.ndarray):
                                assert np.array_equal(u,v)
                            else:
                                assert u==v
                else:
                    assert getattr(x,field)==getattr(y,field),field

def _state(scheduler):
    result=_basic_state(scheduler)
    result.update(finished={k:sorted(v) for k,v in scheduler.finished_req_ids_dict.items()},
                  extras=[(r.request_id,r.num_stale_output_tokens,r.stop_reason,r.num_nans_in_logits,repr(r.events))
                          for r in scheduler.running],
                  processed=getattr(scheduler,'processed_step_seq',None),
                  scheduled=scheduler.sched_step_seq,
                  deferred=[(n,[b.block_id for b in blocks]) for n,blocks in getattr(scheduler,'deferred_frees',[])])
    return result


@pytest.fixture
def native_methods(monkeypatch):
    module=importlib.reload(serving_decode)
    schedule,output=Scheduler.schedule,Scheduler.update_from_output
    allocate,cache=KVCacheManager.allocate_slots,HybridKVCacheCoordinator.cache_blocks
    monkeypatch.setattr(Scheduler,'schedule',schedule)
    monkeypatch.setattr(Scheduler,'update_from_output',output)
    return module,schedule,output,allocate,cache


def test_default_off_preserves_native(native_methods,monkeypatch):
    module,schedule,output,allocate,cache=native_methods
    monkeypatch.delenv('RLFORGE_SERVING_DECODE',raising=False)
    assert module.register() is False
    assert Scheduler.schedule is schedule and Scheduler.update_from_output is output
    assert KVCacheManager.allocate_slots is allocate and HybridKVCacheCoordinator.cache_blocks is cache


def test_changed_native_source_refuses_before_patching(native_methods,monkeypatch):
    module,schedule,output,_,_=native_methods
    monkeypatch.setenv('RLFORGE_SERVING_DECODE','1')
    monkeypatch.setattr(module,'OUTPUT_SOURCE_SHA','0'*64)
    with pytest.raises(RuntimeError,match='source changed'):
        module.install()
    assert Scheduler.schedule is schedule and Scheduler.update_from_output is output


@pytest.mark.parametrize('prompt',[543,544,545,1089])
def test_native_boundary_outputs_and_cache_state(native_methods,monkeypatch,prompt):
    module,schedule,output,allocate,cache=native_methods
    monkeypatch.setenv('RLFORGE_SERVING_DECODE','1');assert module.register()
    left,right=_fixture(prompt,4),_fixture(prompt,4)
    for tick in range(32):
        a,b=schedule(left),right.schedule();assert a==b
        native=output(left,a,_model_output(left,tick))
        candidate=right.update_from_output(b,_model_output(right,tick))
        _compare_output(native,candidate)
        assert _state(left)==_state(right)
        assert 'cache_blocks' not in right.kv_cache_manager.coordinator.__dict__
    assert module.COUNTS['fast_steps']>0 and module.COUNTS['fast_requests']>0
    assert KVCacheManager.allocate_slots is allocate and HybridKVCacheCoordinator.cache_blocks is cache


@pytest.mark.parametrize('mode',['prefill','eos','stop_id','max_tokens','stale','deferred','finished_set'])
def test_mixed_native_fallback_retains_fields(native_methods,monkeypatch,mode):
    module,schedule,output,_,_=native_methods
    monkeypatch.setenv('RLFORGE_SERVING_DECODE','1');module.register()
    left,right=_fixture(1089,8),_fixture(1089,8)
    scheduled=[schedule(left),right.schedule()]
    for scheduler in [left,right]:
        request=scheduler.running[0]
        if mode=='prefill':
            from vllm.v1.metrics.stats import PrefillStats
            request.prefill_stats=PrefillStats()
        elif mode=='eos':
            request.sampling_params.update_from_generation_config({'eos_token_id':8},None)
        elif mode=='stop_id':
            request.sampling_params.stop_token_ids=[8]
        elif mode=='max_tokens':
            request.max_tokens=2
        elif mode=='stale':
            request.num_stale_output_tokens=1;request.drop_stale_output=True
        elif mode=='deferred':
            scheduler.defer_block_free=True;scheduler.processed_step_seq=0;scheduler.sched_step_seq=2
            scheduler.deferred_frees=deque([(4,scheduler.kv_cache_manager.block_pool.get_new_blocks(2))])
        elif mode=='finished_set':
            scheduler.finished_req_ids_dict[7].add('previous')
    a=output(left,scheduled[0],_model_output(left,0));b=right.update_from_output(scheduled[1],_model_output(right,0))
    _compare_output(a,b);assert _state(left)==_state(right)
    assert 'cache_blocks' not in right.kv_cache_manager.coordinator.__dict__


def test_native_exception_restores_cache_override(native_methods,monkeypatch):
    module,_,_,_,_=native_methods
    monkeypatch.setenv('RLFORGE_SERVING_DECODE','1');module.register()
    scheduler=_fixture(1089,4);scheduled=scheduler.schedule();model=_model_output(scheduler,0)
    model.req_id_to_index={}
    with pytest.raises(KeyError):
        scheduler.update_from_output(scheduled,model)
    assert 'cache_blocks' not in scheduler.kv_cache_manager.coordinator.__dict__
