"""GPU inference server for the Qwen arm. Fork of inference_server.py.

Differences:
- reload happens on a background thread (torch.load of the ~2.7GB policy.pt
  takes tens of seconds; the serving loop only applies the ready state_dict,
  a ~2-5s stall, so actors' 10s zmq timeout is never breached);
- the model is QwenQ: it receives the same compact float32 obs bytes as the
  MLP/settf arms and serializes to the text prompt internally.
"""
import argparse
import os
import threading
import time

import numpy as np
import torch
import zmq

from model import build_model, param_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--obs-dim", type=int, required=True)
    ap.add_argument("--addr", required=True)
    ap.add_argument("--batch-cap", type=int, default=64)
    ap.add_argument("--max-wait-ms", type=float, default=2.0)
    ap.add_argument("--decode", choices=["qhead", "pi"], default=None,
                    help="action head for actors; default reads cfg "
                         "'actor_decode' (fallback qhead)")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    K = int(cfg["K"])
    decode = args.decode or cfg.get("actor_decode", "qhead")
    print(f"[infer] decode={decode}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[infer] building model (HF ckpt load, ~1 min)...", flush=True)
    model = build_model(cfg, args.obs_dim).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    policy_path = os.path.join(args.run_dir, "policy.pt")

    # warm every bucketed shape BEFORE binding: actors' zmq timeout is 10s
    # and a cold Triton autotune stall kills the whole actor fleet (attempt 1)
    from model_qwen import warmup
    warmup(model, batches=(args.batch_cap, 16), log=lambda s: print(s, flush=True))

    # ---- background policy reloader ----
    new_state = {}
    lock = threading.Lock()

    def reloader():
        mtime = 0.0
        while True:
            try:
                mt = os.path.getmtime(policy_path)
                if mt > mtime:
                    payload = torch.load(policy_path, map_location="cpu",
                                         weights_only=False)
                    with lock:
                        new_state["payload"] = payload
                    mtime = mt
            except Exception:
                pass
            time.sleep(5)

    threading.Thread(target=reloader, daemon=True).start()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(args.addr)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    print(f"[infer] {cfg.get('name')} params={param_count(model)/1e6:.1f}M "
          f"device={device} addr={args.addr}", flush=True)

    n_req = 0
    t_log = time.time()
    while True:
        with lock:
            payload = new_state.pop("payload", None)
        if payload is not None:
            model.load_state_dict(payload["state_dict"])
            print(f"[infer] reloaded policy @ env_steps="
                  f"{payload.get('env_steps')}", flush=True)

        events = dict(poller.poll(timeout=50))
        if sock not in events:
            continue
        msgs = [sock.recv_multipart()]
        t_end = time.time() + args.max_wait_ms / 1000.0
        while len(msgs) < args.batch_cap:
            remaining = t_end - time.time()
            if remaining <= 0:
                break
            events = dict(poller.poll(timeout=max(0, int(remaining * 1000))))
            if sock not in events:
                break
            msgs.append(sock.recv_multipart())

        obs = np.stack([np.frombuffer(m[-1], dtype=np.float32) for m in msgs])
        x = torch.from_numpy(obs).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=(device == "cuda")):
            if decode == "pi":
                q = model.pi_logits(x).float().cpu().numpy()
            else:
                q = model.q_values(x).float().cpu().numpy()
        for m, row in zip(msgs, q):
            sock.send_multipart([m[0], b"", row.tobytes()])
        n_req += len(msgs)
        if time.time() - t_log > 60:
            print(f"[infer] {n_req / (time.time() - t_log):.0f} req/s "
                  f"last_batch={len(msgs)}", flush=True)
            n_req, t_log = 0, time.time()


if __name__ == "__main__":
    main()
