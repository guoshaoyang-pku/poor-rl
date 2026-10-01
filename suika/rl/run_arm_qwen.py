"""Orchestrate the Qwen arm: DDP learner (torchrun) + infer server + actors + eval.

Fork of run_arm.py. Differences:
- learner launched via torch.distributed.run over cfg learner_gpus;
- waits for the inference server to answer a probe before spawning actors
  (HF ckpt load takes ~1 min; actors would crash on the 10s zmq timeout);
- uses inference_server_qwen.py / evaluator_qwen.py / learner_qwen.py.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def code_fingerprint():
    try:
        rev = subprocess.check_output(
            ["git", "-C", os.path.dirname(_HERE), "rev-parse", "--short", "HEAD"],
            text=True).strip()
        return f"git:{rev}"
    except Exception:
        import hashlib
        h = hashlib.sha256()
        for fn in sorted(os.listdir(_HERE)):
            if fn.endswith(".py"):
                h.update(open(os.path.join(_HERE, fn), "rb").read())
        return f"sha256:{h.hexdigest()[:12]}"


def probe_infer(addr, obs_dim, attempts=40, timeout_s=20.0):
    import zmq
    ctx = zmq.Context.instance()
    for _ in range(attempts):
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(addr)
        try:
            sock.send(np.zeros(obs_dim, dtype=np.float32).tobytes())
            q = np.frombuffer(sock.recv(), dtype=np.float32)
            if q.size > 0:
                sock.close(0)
                return True
        except zmq.ZMQError:
            pass
        finally:
            try:
                sock.close(0)
            except Exception:
                pass
        time.sleep(3.0)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--actors", type=int, default=24)
    ap.add_argument("--obs-dim", type=int, required=True)
    ap.add_argument("--eval-seeds", default="0:16")
    ap.add_argument("--eval-interval-s", type=float, default=900.0)
    ap.add_argument("--master-port", type=int, default=29531)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.run_dir = os.path.abspath(args.run_dir)
    os.makedirs(args.run_dir, exist_ok=True)
    import yaml
    cfg = yaml.safe_load(open(args.config))

    log = open(os.path.join(args.run_dir, "run.log"), "a", buffering=1)
    log.write(f"[code] {code_fingerprint()} config={args.config} "
              f"actors={args.actors} learner_gpus={cfg.get('learner_gpus')} "
              f"infer_gpus={cfg.get('infer_gpus')} time={time.ctime()}\n")

    base_env = dict(os.environ)
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    base_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    base_env["PYTHONPATH"] = _HERE + os.pathsep + base_env.get("PYTHONPATH", "")

    procs = []

    def spawn(name, script, extra, cuda, launcher=None):
        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = cuda
        out = open(os.path.join(args.run_dir, f"{name}.log"), "a", buffering=1)
        cmd = [sys.executable] + (launcher or []) + [
            os.path.join(_HERE, script),
            "--config", args.config, "--run-dir", args.run_dir,
            "--obs-dim", str(args.obs_dim), *extra]
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
        procs.append(p)
        log.write(f"[spawn] {name} pid={p.pid}\n")
        return p

    learner_gpus = cfg.get("learner_gpus", ["0"])
    nproc = len(learner_gpus)
    launcher = []
    if nproc > 1:
        launcher = ["-m", "torch.distributed.run",
                    "--nproc_per_node", str(nproc),
                    "--master_port", str(args.master_port)]
    spawn("learner", "learner_qwen.py", [], ",".join(str(g) for g in learner_gpus),
          launcher=launcher)
    time.sleep(3)

    infer_addrs = []
    if cfg.get("infer") == "server":
        stem = os.path.basename(args.run_dir)
        for gi, gpu in enumerate(cfg.get("infer_gpus", ["0"])):
            addr = f"ipc:///tmp/suika_infer_{stem}_{gi}.ipc"
            spawn(f"infer_server{gi}", "inference_server_qwen.py",
                  ["--addr", addr], str(gpu))
            infer_addrs.append(addr)

    if not probe_infer(infer_addrs[0] if infer_addrs else "",
                       args.obs_dim if infer_addrs else 1):
        log.write("[fatal] inference server never became ready\n")
        for p in procs:
            p.kill()
        sys.exit(1)
    log.write("[probe] inference server ready\n")

    for i in range(args.actors):
        if infer_addrs:
            base_env["SUIKA_INFER_ADDR"] = infer_addrs[i % len(infer_addrs)]
        spawn(f"actor{i}", "actor.py",
              ["--actor-idx", str(i), "--n-actors", str(args.actors)], "")
    spawn("evaluator", "evaluator_qwen.py",
          ["--seeds", args.eval_seeds,
           "--interval-s", str(args.eval_interval_s)],
          str(cfg.get("evaluator_gpu", "0")))

    def shutdown(*_):
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(3)
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    n_procs = len(procs)                    # 1 learner + n_inf + n_actors + 1 eval
    actor_slice = (1 + len(infer_addrs), n_procs - 1)   # actors live here
    dead_logged = set()
    while True:
        rc = procs[0].poll()
        if rc is not None:
            log.write(f"[exit] learner rc={rc}, shutting down arm\n")
            shutdown()
        # actor watchdog: actors can die (e.g. infer timeout) without taking
        # the arm down — surface it, and stop the arm if the fleet is gone
        actors_alive = 0
        for i in range(*actor_slice):
            if procs[i].poll() is None:
                actors_alive += 1
            elif i not in dead_logged:
                dead_logged.add(i)
                log.write(f"[dead] actor#{i - actor_slice[0]} "
                          f"rc={procs[i].returncode}\n")
        if actors_alive == 0:
            log.write("[fatal] all actors dead, shutting down arm\n")
            shutdown()
        time.sleep(10)


if __name__ == "__main__":
    main()
