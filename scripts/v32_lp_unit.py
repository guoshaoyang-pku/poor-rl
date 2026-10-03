"""fast_logprob vs TRL _ChunkedLogProbFunction: values, grads, speed. Real ckpt head + random-ish hidden."""
import os, sys, time, torch
from trl.trainer.utils import _ChunkedLogProbFunction
from safetensors import safe_open
ck = os.environ["GATE_CKPT"]
with safe_open(ck + "/model.safetensors", "pt") as f:
    k = [x for x in f.keys() if "embed_tokens" in x or "lm_head" in x][0]
    W = f.get_tensor(k).float().cuda().requires_grad_(True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
torch.manual_seed(0)
h0 = (torch.randn(N, W.shape[1], device="cuda") * 1.0).requires_grad_(True)
tg = torch.randint(0, W.shape[0], (N,), device="cuda")
gl = torch.randn(N, device="cuda") * 1e-3
ge = torch.randn(N, device="cuda") * 1e-4
def run(fn):
    W.grad = None; h0.grad = None
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp, ent = fn()
    torch.cuda.synchronize(); tf = time.perf_counter() - t
    (lp * gl).sum().backward()
    torch.cuda.synchronize(); tb = time.perf_counter() - t - tf
    return lp.detach(), ent.detach(), h0.grad.clone(), W.grad.clone(), tf, tb
trl = lambda: _ChunkedLogProbFunction.apply(h0, W, None, tg, 1.0, 8192, None, 1.0)
os.environ["RLFORGE_LOGPROB_BWD"] = os.environ.get("RLFORGE_LOGPROB_BWD", "tf32")
import rlforge.fast_logprob as F
fast = lambda: F.logprob_entropy(h0, W, None, tg, 1.0, 1.0)
for _ in range(2): run(trl); run(fast)
a = run(trl); b = run(fast)
cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0))
print(f"N={N} mode={F.BWD_MODE} lp maxdiff {float((a[0]-b[0]).abs().max()):.2e} ent maxdiff {float((a[1]-b[1]).abs().max()):.2e}")
print(f"grad_h cos {cos(a[2],b[2]):.7f} norm ratio {float(b[2].norm()/a[2].norm()):.6f} | grad_W cos {cos(a[3],b[3]):.7f} ratio {float(b[3].norm()/a[3].norm()):.6f}")
print(f"TRL fwd {a[4]:.3f}s bwd {a[5]:.3f}s | fast fwd {b[4]:.3f}s bwd {b[5]:.3f}s | speedup fwd {a[4]/b[4]:.2f}x bwd {a[5]/b[5]:.2f}x")
