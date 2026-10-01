"""Orchestrate one training arm: learner + N actors + periodic evaluator."""
import argparse
import os
import signal
import subprocess
import sys
import time

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--actors", type=int, default=20)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--eval-gpu", default=None,
                    help="GPU for the evaluator (default: share --gpu)")
    ap.add_argument("--obs-dim", type=int, required=True)
    ap.add_argument("--eval-seeds", default="0:16")
    ap.add_argument("--eval-interval-s", type=float, default=300.0)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.run_dir = os.path.abspath(args.run_dir)
    os.makedirs(args.run_dir, exist_ok=True)
    import yaml
    cfg = yaml.safe_load(open(args.config))

    log = open(os.path.join(args.run_dir, "run.log"), "a", buffering=1)
    log.write(f"[code] {code_fingerprint()} config={args.config} "
              f"actors={args.actors} gpu={args.gpu} "
              f"killy={os.environ.get('SUIKA_KILLY', 'stock')} "
              f"time={time.ctime()}\n")

    base_env = dict(os.environ)
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    base_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    base_env["PYTHONPATH"] = _HERE + os.pathsep + base_env.get("PYTHONPATH", "")

    procs = []

    def spawn(name, script, extra, cuda):
        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = cuda
        out = open(os.path.join(args.run_dir, f"{name}.log"), "a", buffering=1)
        p = subprocess.Popen(
            [sys.executable, os.path.join(_HERE, script),
             "--config", args.config, "--run-dir", args.run_dir,
             "--obs-dim", str(args.obs_dim), *extra],
            stdout=out, stderr=subprocess.STDOUT, env=env)
        procs.append(p)
        log.write(f"[spawn] {name} pid={p.pid}\n")
        return p

    learner = spawn("learner", "learner.py", [], args.gpu)
    # gate actor/infer-server startup on the learner's first publish so the
    # served policy is never random init weights (wave3b 19:20 incident).
    policy_path = os.path.join(args.run_dir, "policy.pt")
    deadline = time.time() + 900.0
    while not os.path.exists(policy_path):
        if learner.poll() is not None:
            log.write("[fatal] learner exited before first publish; aborting\n")
            learner.terminate()
            sys.exit(1)
        if time.time() > deadline:
            log.write("[fatal] timeout waiting for policy.pt; aborting\n")
            learner.terminate()
            sys.exit(1)
        time.sleep(2)
    log.write("[gate] policy.pt present, spawning infer servers + actors\n")
    infer_addrs = []
    if cfg.get("infer") == "server":
        # one inference server per listed GPU; actors shard across them.
        # extra GPUs beyond args.gpu come from spare cards on the node.
        infer_gpus = cfg.get("infer_gpus", [args.gpu])
        stem = os.path.basename(args.run_dir)
        for gi, gpu in enumerate(infer_gpus):
            addr = f"ipc:///tmp/suika_infer_{stem}_{gi}.ipc"
            spawn(f"infer_server{gi}", "inference_server.py",
                  ["--addr", addr], str(gpu))
            infer_addrs.append(addr)
        time.sleep(3)
    for i in range(args.actors):
        if infer_addrs:
            base_env["SUIKA_INFER_ADDR"] = infer_addrs[i % len(infer_addrs)]
        spawn(f"actor{i}", "actor.py",
              ["--actor-idx", str(i), "--n-actors", str(args.actors)], "")
    spawn("evaluator", "evaluator.py",
          ["--seeds", args.eval_seeds,
           "--interval-s", str(args.eval_interval_s)],
          args.eval_gpu if args.eval_gpu is not None else args.gpu)

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
    while True:
        rc = learner.poll()
        if rc is not None:
            log.write(f"[exit] learner rc={rc}, shutting down arm\n")
            shutdown()
        time.sleep(10)


if __name__ == "__main__":
    main()
