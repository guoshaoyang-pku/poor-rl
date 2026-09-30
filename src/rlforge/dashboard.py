#!/usr/bin/env python3
"""Live metrics dashboard for rlforge runs -- localhost panel, no external deps.

Serves a single-page panel that auto-refreshes every few seconds and redraws the
moment a new training step lands in the trainer log. Multi-run: every run under
the project root is registered automatically (the "trail"), newest-active first;
finished runs keep their full curves.

    python -m rlforge.dashboard --root /path/to/project --port 8871

    # or a single run explicitly:
    python -m rlforge.dashboard --run runs/X --trainer-log logs/trainer_X.log ...

Then open http://localhost:8871 (on a remote node: ssh -L 8871:localhost:8871 <host>).

Run discovery (--root): scans runs/*/, matching logs/trainer_dp_<suffix>.log and
evals/<name>/history.jsonl by the async_dp_<suffix> naming convention. A registry
file (--registry, JSON, re-read every poll) can add/override entries, e.g. base
eval paths:

    {"async_dp_v3full": {"base_eval": "evals/base_eval300_v3.json"}}

Runs whose files yield no parseable data are hidden from the trail.
Parsing is shared with rlforge.report and cached per run by file mtime.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rlforge.report import parse_eval_history, parse_task_split, parse_trainer_log

_PROGRESS_RE = re.compile(r"(\d+)/(\d+) \[")
_LIVE_WINDOW_S = 600  # trainer log touched within 10 min -> run counts as live


class RunSpec:
    def __init__(self, name, run_dir, trainer_log=None, eval_history=None, base_eval=None):
        self.name = name
        self.run_dir = Path(run_dir)
        self.trainer_log = Path(trainer_log) if trainer_log else None
        self.eval_history = Path(eval_history) if eval_history else None
        self.base_eval = Path(base_eval) if base_eval else None

    def resolve(self, root: Path):
        for attr in ("run_dir", "trainer_log", "eval_history", "base_eval"):
            p = getattr(self, attr)
            if p is not None and not p.is_absolute():
                setattr(self, attr, root / p)
        return self

    def last_touch(self) -> float:
        mt = 0.0
        for p in (self.trainer_log, self.run_dir / "task_split.jsonl", self.eval_history):
            try:
                if p:
                    mt = max(mt, p.stat().st_mtime)
            except OSError:
                pass
        return mt


def discover(root: Path) -> dict[str, RunSpec]:
    found = {}
    runs = root / "runs"
    if runs.exists():
        for d in sorted(runs.iterdir()):
            if not d.is_dir():
                continue
            suffix = d.name[len("async_dp_"):] if d.name.startswith("async_dp_") else d.name
            tlog = root / "logs" / f"trainer_dp_{suffix}.log"
            hist = root / "evals" / d.name / "history.jsonl"
            has_any = (d / "task_split.jsonl").exists() or tlog.exists() or hist.exists()
            if has_any:
                found[d.name] = RunSpec(
                    d.name, d,
                    tlog if tlog.exists() else None,
                    hist if hist.exists() else None,
                )
    return found


class State:
    def __init__(self, args):
        self.root = Path(args.root).resolve() if args.root else None
        self.registry_path = Path(args.registry) if args.registry else None
        self._specs = {}
        self._cache = {}  # (run_name, key) -> (mtime, value)
        if args.run:
            spec = RunSpec(Path(args.run).name, args.run, args.trainer_log,
                           args.eval_history, args.base_eval)
            if self.root:
                spec.resolve(self.root)
            self._specs[spec.name] = spec

    def _refresh_specs(self):
        if self.root:
            for name, spec in discover(self.root).items():
                self._specs.setdefault(name, spec)
        if self.registry_path and self.registry_path.exists():
            try:
                reg = json.load(open(self.registry_path))
            except (OSError, json.JSONDecodeError):
                reg = {}
            for name, entry in reg.items():
                base = self._specs.get(name) or RunSpec(
                    name, entry.get("run", f"runs/{name}"))
                for k, attr in (("trainer_log", "trainer_log"),
                                ("eval_history", "eval_history"),
                                ("base_eval", "base_eval")):
                    if entry.get(k):
                        setattr(base, attr, Path(entry[k]))
                self._specs[name] = base.resolve(self.root) if self.root else base

    @staticmethod
    def _mtime(p: Path | None) -> float:
        try:
            return p.stat().st_mtime if p else -1.0
        except OSError:
            return -1.0

    def _cached(self, run: str, key: str, path: Path | None, builder):
        mt = self._mtime(path)
        ck = (run, key)
        ent = self._cache.get(ck)
        if ent and ent[0] == mt:
            return ent[1]
        val = builder()
        self._cache[ck] = (mt, val)
        return val

    def _progress(self, spec: RunSpec):
        cur = total = None
        if spec.trainer_log and spec.trainer_log.exists():
            try:
                tail = spec.trainer_log.read_bytes()[-262144:].decode(errors="replace")
                prog = _PROGRESS_RE.findall(tail)
                if prog:
                    cur, total = int(prog[-1][0]), int(prog[-1][1])
            except OSError:
                pass
        return cur, total

    def list_runs(self) -> list[dict]:
        self._refresh_specs()
        out = []
        for name, spec in self._specs.items():
            cur, total = self._progress(spec)
            touch = spec.last_touch()
            has_data = bool(touch)
            if cur is not None or (spec.eval_history and spec.eval_history.exists()):
                has_data = True
            if not has_data:
                continue  # no curves -> skip (per user request)
            live = cur is not None and (total is None or cur < total) and \
                (time.time() - touch) < _LIVE_WINDOW_S
            out.append({"name": name, "step": cur, "total": total,
                        "live": live, "touch": touch})
        out.sort(key=lambda r: (-r["live"], -r["touch"]))
        return out

    def payload(self, name: str) -> dict:
        self._refresh_specs()
        spec = self._specs.get(name)
        if spec is None:
            return {"error": f"unknown run {name!r}", "runs": [r["name"] for r in self.list_runs()]}
        steps = self._cached(name, "steps", spec.trainer_log,
                             lambda: parse_trainer_log(spec.trainer_log)
                             if spec.trainer_log and spec.trainer_log.exists() else [])
        task_path = spec.run_dir / "task_split.jsonl"
        tasks = self._cached(name, "tasks", task_path,
                             lambda: parse_task_split(task_path) if task_path.exists() else [])
        evals = self._cached(name, "evals", spec.eval_history,
                             lambda: parse_eval_history(spec.eval_history)
                             if spec.eval_history and spec.eval_history.exists() else [])
        base = self._cached(name, "base", spec.base_eval, lambda: self._load_base(spec))
        return {"run": name, "steps": steps, "task_bins": tasks,
                "evals": evals, "base": base, "status": self._status(spec, steps, evals)}

    @staticmethod
    def _load_base(spec):
        if not spec.base_eval or not spec.base_eval.exists():
            return None
        d = json.load(open(spec.base_eval))
        if "overall" not in d and "summary" in d:
            d = d["summary"]
        return d

    def _status(self, spec, steps, evals) -> dict:
        cur, total = self._progress(spec)
        recent = [s["step_s"] for s in steps[-20:] if s.get("step_s")]
        sps = sum(recent) / len(recent) if recent else None
        touch = spec.last_touch()
        live = cur is not None and (total is None or cur < total) and \
            (time.time() - touch) < _LIVE_WINDOW_S
        return {
            "step": cur if cur is not None else len(steps),
            "total": total,
            "step_s": round(sps, 1) if sps else None,
            "eta_min": round((total - cur) * sps / 60) if (total and cur and sps and live) else None,
            "n_evals": len(evals),
            "live": live,
            "updated": time.strftime("%H:%M:%S"),
        }


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>rlforge dashboard</title>
<style>
 body{font-family:-apple-system,'Helvetica Neue',sans-serif;margin:0;background:#fafafa;color:#222}
 header{padding:10px 18px;background:#1f2733;color:#fff;display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
 header h1{font-size:16px;margin:0}
 header .stat{font-size:12px;opacity:.9}
 header .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#4caf50;margin-right:5px}
 header .dot.dead{background:#888}
 #runsel{background:#2c3949;color:#fff;border:1px solid #4a5a6e;border-radius:4px;padding:2px 6px;font-size:12.5px}
 #grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:10px;padding:12px}
 .panel{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px}
 .panel h3{margin:2px 4px 6px;font-size:12.5px;color:#444;font-weight:600}
 canvas{width:100%;height:220px;display:block}
</style></head><body>
<header><h1>rlforge</h1><select id="runsel"></select><span class="stat" id="status">loading…</span></header>
<div id="grid"></div>
<script>
const PALETTE = ['#1f77b4','#2ca02c','#d62728','#9467bd','#ff7f0e','#17becf','#8c564b','#e377c2'];
let colorIdx = 0; const colorFor = {};
let currentRun = localStorage.getItem('rlforge_run') || null;
function col(name){ if(!(name in colorFor)){ colorFor[name]=PALETTE[colorIdx++%PALETTE.length]; } return colorFor[name]; }
function rolling(ys,w){ const out=[];let s=0;const q=[]; for(const y of ys){ if(y==null){out.push(null);continue;} q.push(y);s+=y; if(q.length>w)s-=q.shift(); out.push(s/q.length);} return out; }
function panel(title){ const d=document.createElement('div'); d.className='panel'; const h=document.createElement('h3'); h.textContent=title; d.appendChild(h);
  const c=document.createElement('canvas'); d.appendChild(c); document.getElementById('grid').appendChild(d); return c; }
function draw(cv, cfg){
  const dpr = window.devicePixelRatio||1;
  const W = cv.clientWidth, H = cv.clientHeight;
  cv.width=W*dpr; cv.height=H*dpr;
  const g = cv.getContext('2d'); g.scale(dpr,dpr); g.clearRect(0,0,W,H);
  const m={l:46,r:8,t:6,b:20}, pw=W-m.l-m.r, ph=H-m.t-m.b;
  let xs=[], ys=[];
  for(const s of cfg.series){ xs=xs.concat(s.x); ys=ys.concat(s.y.filter(v=>v!=null)); }
  if(!xs.length||!ys.length){ g.fillStyle='#999'; g.font='11px sans-serif'; g.fillText('no data yet', m.l+8, m.t+16); return; }
  let xmin=Math.min(...xs), xmax=Math.max(...xs);
  let ymin=cfg.ymin!=null?cfg.ymin:Math.min(...ys), ymax=cfg.ymax!=null?cfg.ymax:Math.max(...ys);
  if(cfg.logy){ ymin=Math.log10(Math.max(ymin,1e-6)); ymax=Math.log10(Math.max(ymax,1e-6)); }
  if(ymax-ymin<1e-9){ ymax=ymin+1; }
  const pad=(ymax-ymin)*0.06; if(cfg.ymin==null)ymin-=pad; if(cfg.ymax==null)ymax+=pad;
  const tx=v=>m.l+(xmax>xmin?(v-xmin)/(xmax-xmin):0.5)*pw;
  const ty=v=>{ if(cfg.logy)v=Math.log10(Math.max(v,1e-6)); return m.t+ph-(v-ymin)/(ymax-ymin)*ph; };
  g.strokeStyle='#e6e6e6'; g.fillStyle='#888'; g.font='10px sans-serif'; g.lineWidth=1;
  for(let i=0;i<=4;i++){ const yv=ymin+(ymax-ymin)*i/4, py=m.t+ph-ph*i/4;
    g.beginPath(); g.moveTo(m.l,py); g.lineTo(m.l+pw,py); g.stroke();
    const lab=cfg.logy?Math.pow(10,yv).toPrecision(2):yv.toPrecision(3);
    g.fillText(lab, 4, py+3); }
  for(let i=0;i<=5;i++){ const xv=xmin+(xmax-xmin)*i/5, px=tx(xv);
    g.fillText(cfg.xfmt?cfg.xfmt(xv):xv.toFixed(0), px-8, H-6); }
  for(const hl of (cfg.hlines||[])){ g.strokeStyle=hl.color; g.setLineDash([4,3]);
    g.beginPath(); g.moveTo(m.l,ty(hl.y)); g.lineTo(m.l+pw,ty(hl.y)); g.stroke(); g.setLineDash([]); }
  cfg.series.forEach((s,si)=>{
    g.strokeStyle=s.color||col(s.name); g.lineWidth=s.thin?1:1.8; g.globalAlpha=s.alpha!=null?s.alpha:1;
    g.beginPath(); let pen=false;
    for(let i=0;i<s.x.length;i++){ const v=s.y[i]; if(v==null){pen=false;continue;}
      const px=tx(s.x[i]), py=ty(v); if(!pen){g.moveTo(px,py);pen=true;} else g.lineTo(px,py); }
    g.stroke();
    if(s.dots){ g.fillStyle=s.color||col(s.name); for(let i=0;i<s.x.length;i++){ const v=s.y[i]; if(v==null)continue;
      g.beginPath(); g.arc(tx(s.x[i]),ty(v),2.6,0,7); g.fill(); } }
    g.globalAlpha=1;
  });
  g.font='10px sans-serif'; let lx=m.l+4;
  for(const s of cfg.series){ g.fillStyle=s.color||col(s.name); g.fillRect(lx,m.t+2,8,3);
    g.fillStyle='#555'; g.fillText(s.name, lx+11, m.t+7); lx+=11+g.measureText(s.name).width+14; }
}
function series(xs, ys, name, opts){ return Object.assign({x:xs, y:ys, name:name, color:col(name)}, opts||{}); }

async function refreshRuns(){
  let runs; try{ runs = await (await fetch('/api/runs')).json(); } catch(e){ return; }
  const sel = document.getElementById('runsel');
  const prev = currentRun;
  sel.innerHTML = '';
  for(const r of runs){
    const o = document.createElement('option'); o.value = r.name;
    o.textContent = (r.live?'● ':'○ ')+r.name+(r.step!=null?('  '+r.step+(r.total?'/'+r.total:'')):'');
    sel.appendChild(o);
  }
  if(!prev || !runs.some(r=>r.name===prev)) currentRun = runs.length ? runs[0].name : null;
  sel.value = currentRun;
}
document.getElementById('runsel').addEventListener('change', e=>{
  currentRun = e.target.value; localStorage.setItem('rlforge_run', currentRun); refreshMetrics();
});

async function refreshMetrics(){
  if(!currentRun) return;
  let d; try{ d = await (await fetch('/api/metrics?run='+encodeURIComponent(currentRun))).json(); } catch(e){ return; }
  if(d.error){ return; }
  const st=d.status;
  document.getElementById('status').innerHTML =
    '<span class="dot'+(st.live?'':' dead')+'"></span>'+(st.live?'live · ':'finished · ')+
    'step '+(st.step!=null?st.step:'?')+(st.total?' / '+st.total:'')+
    (st.step_s&&st.live?' · '+st.step_s+' s/step':'')+(st.eta_min!=null?' · ETA '+st.eta_min+' min':'')+
    ' · '+st.n_evals+' held-out evals · updated '+st.updated;
  document.getElementById('grid').innerHTML=''; colorIdx=0; for(const k in colorFor)delete colorFor[k];

  const S=d.steps, sx=S.map(r=>r.step);
  if(S.length){
    const rw=S.map(r=>r.reward);
    draw(panel('reward per step (raw + 20-step mean)'), {series:[
      series(sx, rw, 'reward', {thin:true, alpha:0.2}),
      series(sx, rolling(rw,20), 'reward (smooth)')], ymin:-2.1, ymax:1.1});
    draw(panel('clip ratios: completion truncation & GSPO seq_clip_low'), {series:[
      series(sx, S.map(r=>r.trunc), 'truncation frac'),
      series(sx, S.map(r=>r.seq_clip_low), 'gspo seq_clip_low')],
      ymin:0, ymax:1, hlines:[{y:0.5,color:'#d62728'},{y:0.9,color:'#8b0000'}]});
    draw(panel('sequence length (mean completion tokens, log)'), {series:[
      series(sx, S.map(r=>r.meanlen), 'mean tokens')], logy:true});
    draw(panel('KL & entropy'), {series:[
      series(sx, S.map(r=>r.kl), 'kl'), series(sx, S.map(r=>r.entropy), 'entropy')]});
  }
  const T=d.task_bins, tx=T.map(r=>r.hour);
  if(T.length){
    draw(panel('per-task train reward (exact reconstruction, incl. penalties)'), {series:[
      series(tx, T.map(r=>r.mcq_reward), 'MCQ'), series(tx, T.map(r=>r.rank_reward), 'ranking')],
      xfmt:v=>v.toFixed(1)+'h'});
    draw(panel('per-task train accuracy (correctness only)'), {series:[
      series(tx, T.map(r=>r.mcq_correct), 'MCQ correct'),
      series(tx, T.map(r=>r.rank_exact), 'ranking exact'),
      series(tx, T.map(r=>r.rank_score), 'ranking partial', {thin:true})],
      ymin:0, ymax:1.02, xfmt:v=>v.toFixed(1)+'h'});
    draw(panel('per-task truncation'), {series:[
      series(tx, T.map(r=>r.mcq_trunc), 'MCQ'), series(tx, T.map(r=>r.rank_trunc), 'ranking')],
      ymin:0, ymax:1, xfmt:v=>v.toFixed(1)+'h'});
    const srcs=[...new Set(T.flatMap(r=>Object.keys(r.by_source||{})))];
    if(srcs.length){
      const get=(r,s,k)=>{const v=(r.by_source||{})[s]; return v?v[k]:null;};
      draw(panel('per-source train reward'), {series:srcs.map(s=>series(tx,T.map(r=>get(r,s,'reward')),s)),
        xfmt:v=>v.toFixed(1)+'h'});
      draw(panel('per-source train accuracy'), {series:srcs.map(s=>series(tx,T.map(r=>get(r,s,'correct')),s)),
        ymin:0, ymax:1.02, xfmt:v=>v.toFixed(1)+'h'});
      draw(panel('per-source truncation'), {series:srcs.map(s=>series(tx,T.map(r=>get(r,s,'trunc')),s)),
        ymin:0, ymax:1, xfmt:v=>v.toFixed(1)+'h'});
    }
  }
  const E=d.evals;
  if(E.length || d.base){
    const ex=E.map(e=>e.step);
    const ser=[];
    if(d.base){ ser.push({x:[0],y:[d.base.overall.accuracy],name:'base (step 0)',color:'#d62728',dots:true}); }
    ser.push(series(ex, E.map(e=>e.overall), 'held-out overall', {dots:true}));
    ser.push(series(ex, E.map(e=>e.mcq), 'held-out MCQ', {dots:true}));
    if(E.some(e=>e.ranking!=null)) ser.push(series(ex, E.map(e=>e.ranking), 'held-out ranking', {dots:true}));
    const hl=[];
    const cm=E.map(e=>e.const_mcq).find(v=>v!=null), cr=E.map(e=>e.const_rank).find(v=>v!=null);
    if(cm!=null)hl.push({y:cm,color:'#1f77b4'}); if(cr!=null)hl.push({y:cr,color:'#2ca02c'});
    draw(panel('held-out accuracy ladder'), {series:ser, ymin:0, ymax:1.02, hlines:hl});
  }
}
async function tick(){ await refreshRuns(); await refreshMetrics(); }
tick(); setInterval(tick, 10000);
</script></body></html>"""


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/metrics"):
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(self.path).query)
                name = (q.get("run") or [None])[0]
                if name is None:
                    runs = state.list_runs()
                    name = runs[0]["name"] if runs else ""
                try:
                    body = json.dumps(state.payload(name)).encode()
                except Exception as e:  # a half-written file must never kill the panel
                    body = json.dumps({"error": str(e)}).encode()
                self._reply(200, body, "application/json")
            elif self.path.startswith("/api/runs"):
                self._reply(200, json.dumps(state.list_runs()).encode(), "application/json")
            elif self.path in ("/", "/index.html"):
                self._reply(200, _HTML.encode(), "text/html; charset=utf-8")
            else:
                self._reply(404, b"not found", "text/plain")

        def _reply(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="project root with runs/ logs/ evals/; auto-registers every run")
    ap.add_argument("--registry", default=None,
                    help="optional JSON registry for per-run overrides (re-read every poll)")
    ap.add_argument("--run", default=None, help="single-run mode: run directory")
    ap.add_argument("--trainer-log", default=None)
    ap.add_argument("--eval-history", default=None)
    ap.add_argument("--base-eval", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8871)
    args = ap.parse_args()
    if not args.root and not args.run:
        ap.error("either --root (multi-run trail) or --run (single run) is required")
    state = State(args)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[dashboard] http://{args.host}:{args.port} "
          f"(remote: ssh -L {args.port}:localhost:{args.port} <host>)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
