#!/usr/bin/env python3
"""Live metrics dashboard for rlforge runs -- localhost panel, no external deps.

Serves a single-page panel that auto-refreshes every few seconds and redraws the
moment a new training step lands in the trainer log:

    python -m rlforge.dashboard --run runs/async_dp_flip450 \
        --trainer-log logs/trainer_dp_flip450.log \
        --eval-history evals/async_dp_flip450/history.jsonl \
        --base-eval evals/base_flip50.json --port 8871

Then open http://localhost:8871 (on a remote node: ssh -L 8871:localhost:8871 <host>).

Panels: per-step reward / truncation & GSPO seq-clip / per-task reward-accuracy-
truncation / per-source curves / sequence length / KL & entropy / held-out ladder.
Parsing is shared with rlforge.report and cached by file mtime, so polling is cheap.
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


class State:
    def __init__(self, args):
        self.run = Path(args.run)
        self.trainer_log = Path(args.trainer_log)
        self.eval_history = Path(args.eval_history) if args.eval_history else None
        self.base_eval_path = Path(args.base_eval) if args.base_eval else None
        self._cache = {}

    @staticmethod
    def _mtime(p: Path | None) -> float:
        try:
            return p.stat().st_mtime if p else -1.0
        except OSError:
            return -1.0

    def _cached(self, key: str, path: Path | None, builder):
        mt = self._mtime(path)
        ent = self._cache.get(key)
        if ent and ent[0] == mt:
            return ent[1]
        val = builder()
        self._cache[key] = (mt, val)
        return val

    def payload(self) -> dict:
        steps = self._cached("steps", self.trainer_log,
                             lambda: parse_trainer_log(self.trainer_log) if self.trainer_log.exists() else [])
        task_path = self.run / "task_split.jsonl"
        tasks = self._cached("tasks", task_path,
                             lambda: parse_task_split(task_path) if task_path.exists() else [])
        evals = self._cached("evals", self.eval_history,
                             lambda: parse_eval_history(self.eval_history)
                             if self.eval_history and self.eval_history.exists() else [])
        base = self._cached("base", self.base_eval_path, self._load_base)
        return {"run": self.run.name, "steps": steps, "task_bins": tasks,
                "evals": evals, "base": base, "status": self._status(steps, evals)}

    def _load_base(self):
        if not self.base_eval_path or not self.base_eval_path.exists():
            return None
        d = json.load(open(self.base_eval_path))
        if "overall" not in d and "summary" in d:
            d = d["summary"]
        return d

    def _status(self, steps, evals) -> dict:
        cur = total = None
        try:
            tail = self.trainer_log.read_bytes()[-262144:].decode(errors="replace")
            prog = _PROGRESS_RE.findall(tail)
            if prog:
                cur, total = int(prog[-1][0]), int(prog[-1][1])
        except OSError:
            pass
        recent = [s["step_s"] for s in steps[-20:] if s.get("step_s")]
        sps = sum(recent) / len(recent) if recent else None
        return {
            "step": cur if cur is not None else len(steps),
            "total": total,
            "step_s": round(sps, 1) if sps else None,
            "eta_min": round((total - cur) * sps / 60) if (total and cur and sps) else None,
            "n_evals": len(evals),
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
 #grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:10px;padding:12px}
 .panel{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px}
 .panel h3{margin:2px 4px 6px;font-size:12.5px;color:#444;font-weight:600}
 canvas{width:100%;height:220px;display:block}
</style></head><body>
<header><h1 id="title">rlforge</h1><span class="stat" id="status">loading…</span></header>
<div id="grid"></div>
<script>
const PALETTE = ['#1f77b4','#2ca02c','#d62728','#9467bd','#ff7f0e','#17becf','#8c564b','#e377c2'];
let colorIdx = 0; const colorFor = {};
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

async function refresh(){
  let d; try{ d = await (await fetch('/api/metrics')).json(); } catch(e){ return; }
  document.getElementById('title').textContent = 'rlforge · '+d.run;
  const st=d.status;
  document.getElementById('status').innerHTML =
    '<span class="dot"></span>step '+(st.step!=null?st.step:'?')+(st.total?' / '+st.total:'')+
    (st.step_s?' · '+st.step_s+' s/step':'')+(st.eta_min!=null?' · ETA '+st.eta_min+' min':'')+
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
refresh(); setInterval(refresh, 10000);
</script></body></html>"""


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/metrics"):
                try:
                    body = json.dumps(state.payload()).encode()
                except Exception as e:  # a half-written file must never kill the panel
                    body = json.dumps({"error": str(e)}).encode()
                self._reply(200, body, "application/json")
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--trainer-log", required=True)
    ap.add_argument("--eval-history", default=None)
    ap.add_argument("--base-eval", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8871)
    args = ap.parse_args()
    state = State(args)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[dashboard] {state.run.name} -> http://{args.host}:{args.port} "
          f"(remote: ssh -L {args.port}:localhost:{args.port} <host>)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
