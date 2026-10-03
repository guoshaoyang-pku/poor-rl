"""fused_ops vs transformers originals on real activation shapes: values + grads."""
import torch, torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
import rlforge.fused_ops as fo
torch.manual_seed(0)
B, C, L = 8, 6144, 3000
x = torch.randn(B, C, L, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
w = (torch.randn(C, 4, device="cuda") * 0.3).requires_grad_(True)
def go(fn):
    x.grad = None; w.grad = None
    y = fn(x, w, None, activation="silu"); g = torch.randn_like(y)
    (y.float() * g.float()).sum().backward()
    return y.detach(), x.grad.clone(), w.grad.clone()
torch.manual_seed(1); a = go(mq.causal_conv1d_fn)
torch.manual_seed(1); b = go(fo.causal_conv1d_fn)
print("conv y maxdiff", float((a[0].float()-b[0].float()).abs().max()), "exact frac", float((a[0]==b[0]).float().mean()))
print("conv dx cos", float(F.cosine_similarity(a[1].float().flatten(), b[1].float().flatten(), dim=0)), "dw cos", float(F.cosine_similarity(a[2].flatten(), b[2].flatten(), dim=0)))
import time
for fn, n in [(mq.causal_conv1d_fn, "orig"), (fo.causal_conv1d_fn, "fused")]:
    for _ in range(3): go(fn)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(5): go(fn)
    torch.cuda.synchronize(); print(n, "conv fwd+bwd ms", (time.perf_counter()-t)/5*1e3)
# rmsnorm
n = mq.Qwen3_5RMSNorm(1024).cuda(); n.weight.data.normal_(0, 0.1)
h = torch.randn(64000, 1024, device="cuda").requires_grad_(True)
def go2(f):
    h.grad = None; n.weight.grad = None
    y = f(h); y.sum().backward(); return y.detach(), h.grad.clone(), n.weight.grad.clone()
orig = mq.Qwen3_5RMSNorm.forward
a = go2(n)
fo.install("norm")
b = go2(n)
print("rms y maxdiff", float((a[0]-b[0]).abs().max()), "dh maxdiff", float((a[1]-b[1]).abs().max()), "dw rel", float((a[2]-b[2]).norm()/a[2].norm()))
