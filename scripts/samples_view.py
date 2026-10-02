"""Render rollout/eval sample JSONL files as a standalone HTML page for the hub.

Data layout: artifacts/samples/<run_id>/
  meta.json            run metadata (run.json content; all fields optional)
  train.jsonl          rollout samples from the trainer sampler (AIQ_SAMPLE_LOG)
  eval_<label>.jsonl   up to 10 display samples per held-out eval
"""
from __future__ import annotations

import html
import json
from pathlib import Path

SAMPLES_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "samples"

# seconds per 5-step display window; training runs ~21-25 s/step
WINDOW_S = 120
PER_WINDOW = 10


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _badge(reward) -> str:
    try:
        r = float(reward)
    except (TypeError, ValueError):
        return '<span class="b gray">reward ?</span>'
    cls = "green" if r >= 0.99 else ("amber" if r >= 0 else "red")
    return f'<span class="b {cls}">reward {r:+.2f}</span>'


def _card(sample: dict, eval_mode: bool) -> str:
    task = _esc(sample.get("task", "?"))
    src = _esc(sample.get("source") or sample.get("family") or "?")
    gold = _esc(sample.get("gold", "?"))
    pred = sample.get("pred")
    pred_html = f'<span class="b gray">pred {_esc(pred)}</span>' if pred else ""
    ntok = sample.get("n_tokens") or sample.get("completion_tokens") or "?"
    trunc = '<span class="b red">truncated</span>' if sample.get("truncated") else ""
    qid = f'<span class="b gray">{_esc(sample["question_id"])}</span>' if sample.get("question_id") else ""
    completion = _esc(sample.get("completion", ""))
    return (
        f'<details class="card"><summary>{_badge(sample.get("reward"))}'
        f'<span class="b blue">{task}</span><span class="b gray">{src}</span>'
        f'<span class="b gray">gold {gold}</span>{pred_html}{qid}'
        f'<span class="b gray">{ntok} tok</span>{trunc}</summary>'
        f"<pre>{completion}</pre></details>"
    )


def _pick_window(rows: list[dict]) -> list[dict]:
    """Stratified pick: lowest rewards first (failures are the interesting ones)."""
    rows = sorted(rows, key=lambda r: (float(r.get("reward", 0)), r.get("t", 0)))
    head = rows[: PER_WINDOW // 2]
    tail = rows[-(PER_WINDOW - len(head)):] if len(rows) > PER_WINDOW - len(head) else []
    seen, out = set(), []
    for r in head + tail:
        k = id(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return sorted(out, key=lambda r: r.get("t", 0))


def render_run(run_id: str) -> bytes:
    run_dir = SAMPLES_ROOT / run_id
    if not run_dir.is_dir():
        return b"run not found"
    meta = {}
    if (run_dir / "meta.json").is_file():
        meta = json.loads((run_dir / "meta.json").read_text())
    hp = meta.get("hyperparams", {})
    meta_rows = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>"
        for k, v in [
            ("run_id", meta.get("run_id", run_id)),
            ("node", meta.get("node", "?")),
            ("started", meta.get("started", "?")),
            ("stop", meta.get("stop_reason", "?")),
            ("model", meta.get("model", {}).get("path", "?")),
            ("data", f'{meta.get("data", {}).get("path", "?")} md5={meta.get("data", {}).get("md5", "?")}'),
            ("layout", json.dumps(meta.get("gpu_layout", {}))),
            ("hyper", json.dumps(hp)),
        ]
    )
    parts = [CSS, f"<h1>{_esc(run_id)}</h1><p><a href='/'>⌂ 实验主页</a></p>",
             f"<h2>meta</h2><table class='meta'>{meta_rows}</table>"]

    eval_files = sorted(run_dir.glob("eval_*.jsonl"))
    if eval_files:
        parts.append("<h2>held-out 评测采样（每次评测 10 条）</h2>")
        for ef in eval_files:
            label = ef.stem[len("eval_"):]
            rows = [json.loads(line) for line in ef.read_text().splitlines() if line.strip()]
            acc = [r.get("reward") for r in rows]
            parts.append(f"<h3>{_esc(label)} · {len(rows)} 条</h3>")
            parts.extend(_card(r, True) for r in rows)

    train_file = run_dir / "train.jsonl"
    if train_file.is_file():
        rows = [json.loads(line) for line in train_file.read_text().splitlines() if line.strip()]
        t0 = meta.get("started_ts")
        if not t0 and rows:
            t0 = min(r.get("t", 0) for r in rows)
        windows: dict[int, list[dict]] = {}
        for r in rows:
            w = int((r.get("t", t0) - t0) // WINDOW_S)
            windows.setdefault(w, []).append(r)
        parts.append(f"<h2>训练 rollout 采样（每 ~5 steps 最多 {PER_WINDOW} 条，共 {len(rows)} 条采样）</h2>")
        for w in sorted(windows, reverse=True):
            picked = _pick_window(windows[w])
            rs = [float(x.get("reward", 0)) for x in windows[w]]
            mean_r = sum(rs) / len(rs) if rs else 0
            parts.append(f"<h3>window {w}（step ~{w * 5}–{w * 5 + 4}）· 采样 {len(windows[w])} 条 · 窗口均 reward {mean_r:+.3f}</h3>")
            parts.extend(_card(r, False) for r in picked)

    return ("<!doctype html><html lang=zh><head><meta charset=utf-8>"
            f"<title>{_esc(run_id)} samples</title></head><body>" + "".join(parts) + "</body></html>").encode()


CSS = """<style>
body{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
h1{font-size:20px}h2{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:13px;color:#555;margin:14px 0 6px}
table.meta{border-collapse:collapse;font-size:12px}
table.meta th{text-align:left;padding:2px 10px 2px 0;color:#666;vertical-align:top}
table.meta td{word-break:break-all}
.card{margin:4px 0;border:1px solid #e2e2e2;border-radius:6px}
.card summary{cursor:pointer;padding:6px 10px;list-style:none}
.card summary::-webkit-details-marker{display:none}
.card pre{margin:0;padding:8px 12px;white-space:pre-wrap;word-break:break-word;font-size:12px;background:#fafafa;border-top:1px solid #eee;max-height:400px;overflow:auto}
.b{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;margin-right:6px;color:#fff}
.green{background:#2e7d32}.amber{background:#f9a825}.red{background:#c62828}.gray{background:#9e9e9e}.blue{background:#1565c0}
</style>"""
