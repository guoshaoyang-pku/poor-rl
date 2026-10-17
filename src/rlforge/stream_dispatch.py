"""Dispatch an infinite rollout stream without Accelerate's end-of-data lookahead."""
from accelerate.data_loader import (
    DataLoaderDispatcher, broadcast, find_batch_size, initialize_tensors,
    is_torch_version, send_to_device,
)


class InfiniteRolloutDispatcher(DataLoaderDispatcher):
    """The rollout queue ends by trainer max_steps, never by dataset exhaustion."""

    def __iter__(self):
        self.begin()
        self.set_epoch(self.iteration)
        main_iterator = None
        if is_torch_version(">=", "2.0.1") or self.state.process_index == 0:
            main_iterator = iter(self.base_dataloader)
        self._stop_iteration = False
        batch_index = 0
        try:
            while True:
                batch, info = self._fetch_batches(main_iterator)
                if self._stop_iteration or info[0] is None:
                    raise RuntimeError("infinite rollout dispatcher unexpectedly exhausted")
                if self.state.process_index != 0:
                    batch = initialize_tensors(info[0])
                batch = send_to_device(batch, self.state.device, non_blocking=self._non_blocking)
                batch = broadcast(batch, from_process=0)
                size = find_batch_size(batch)
                if size % self.state.num_processes:
                    raise RuntimeError("rollout planner emitted incomplete rank rows")
                per_rank = size // self.state.num_processes
                rank = self.state.process_index
                batch = self.slice_fn(
                    batch, slice(rank * per_rank, (rank + 1) * per_rank),
                    process_index=rank, num_processes=self.state.num_processes,
                )
                if batch_index >= self.skip_batches:
                    yield batch
                batch_index += 1
        finally:
            self.iteration += 1
            self.end()


def install(trainer):
    """Apply only to this trainer's infinite rollout loader, not evaluation loaders."""
    original = trainer.get_train_dataloader

    def get_train_dataloader(*args, **kwargs):
        loader = original(*args, **kwargs)
        if not isinstance(loader, DataLoaderDispatcher):
            raise TypeError("rollout stream requires Accelerate DataLoaderDispatcher")
        # Accelerate exposes __class__ as a read-only DataLoader compatibility
        # property. Keep the prepared loader's state and base loader unchanged.
        stream = object.__new__(InfiniteRolloutDispatcher)
        stream.__dict__.update(loader.__dict__)
        print("[rlforge] infinite rollout dispatch: end-of-data lookahead disabled", flush=True)
        return stream

    trainer.get_train_dataloader = get_train_dataloader
