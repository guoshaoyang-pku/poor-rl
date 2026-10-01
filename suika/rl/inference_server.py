"""GPU inference server for one arm: batched Q-value requests over zmq.

Actors (thin CPU processes) connect via REQ and send one flat obs
(float32 bytes) per request; replies are K float32 Q-values.
The server batches whatever is queued (up to --batch-cap, waiting at most
--max-wait-ms for the first stragglers), runs one GPU forward, and replies.
Policy weights hot-reload whenever learner publishes a new policy.pt.
"""
import argparse
import os
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
    ap.add_argument("--max-wait-ms", type=float, default=1.0)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    K = int(cfg["K"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, args.obs_dim).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    policy_path = os.path.join(args.run_dir, "policy.pt")
    mtime = 0.0

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(args.addr)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    print(f"[infer] {cfg.get('name')} params={param_count(model)/1e6:.2f}M "
          f"device={device} addr={args.addr}", flush=True)

    n_req = 0
    t_log = time.time()
    while True:
        try:
            mt = os.path.getmtime(policy_path)
            if mt > mtime:
                payload = torch.load(policy_path, map_location="cpu",
                                     weights_only=False)
                model.load_state_dict(payload["state_dict"])
                mtime = mt
                print(f"[infer] reloaded policy @ env_steps="
                      f"{payload.get('env_steps')}", flush=True)
        except Exception:
            pass

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
