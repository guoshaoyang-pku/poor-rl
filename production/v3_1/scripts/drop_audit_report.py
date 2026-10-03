#!/usr/bin/env python3
"""Summarize rlforge drop_audit.jsonl over a step range (rlforge_v3_next, 2026-10-03).

    python scripts/drop_audit_report.py runs/<run>/drop_audit.jsonl [--from 10] [--to 200] [--per-step] [--json]

Every number is recombined from the additive sums in the per-step records (counts, reward sums and counts,
length sums and 512-token histograms), so a range summary is exact, not a mean of per-step means. Percentiles over
a range come from the fine histogram (resolution = fine_bin tokens, reported as the bin midpoint; the truncated
bin is reported as the cap). Definitions are in src/rlforge/drop_audit.py.

A relaunch or resume into the same file appends a new segment (a "header" line, then its steps). Steps are
deduplicated by keeping the LAST record written for each step number, so a resume from a checkpoint replaces the
steps it redoes and keeps the earlier ones. Missing step numbers in the range are reported (the sums then cover
fewer steps than the range).
"""

import argparse
import json
import math
import sys

GROUP_KEYS = (
    "groups", "kept_full", "dropped_full", "partial", "short",
    "samples", "samples_dropped", "samples_dropped_partial",
    "r_kept_full_sum", "r_kept_full_n", "r_dropped_full_sum", "r_dropped_full_n",
    "r_partial_kept_sum", "r_partial_kept_n", "r_partial_dropped_sum", "r_partial_dropped_n",
    "lag_sum", "lag_n",
)
BUCKETINGS = (("buckets", "Groups by LONGEST member (trunc = any member hit the cap)"),
              ("buckets_median", "Groups by MEDIAN member length"),
              ("buckets_ntrunc", "Groups by number of TRUNCATED members (t0 none, t1 one, t2p two or more)"))
SAMPLE_KEYS = ("arrived", "kept", "dropped", "trained")
REWARD_KEYS = ("trained_samples_sum", "trained_samples_n", "kept_samples_sum", "kept_samples_n",
               "dropped_samples_sum", "dropped_samples_n")


def load(path, lo, hi, include_final, info=None):
    """Step records in [lo, hi], last record per step wins; finals (one per segment) only with include_final.
    ``info`` (a dict) receives headers, duplicate and gap diagnostics."""
    by_step, finals, headers, n_dup, n_bad = {}, {}, [], 0, 0
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                print(f"[warn] line {i + 1}: not JSON, skipped", file=sys.stderr)
                continue
            if not isinstance(r, dict):
                n_bad += 1
                continue
            if r.get("type") == "header":
                headers.append(r)
                continue
            if r.get("final"):
                finals[r.get("seg")] = r
                continue
            if r["step"] in by_step:
                n_dup += 1
            by_step[r["step"]] = r
    steps = sorted(s for s in by_step if (lo is None or s >= lo) and (hi is None or s <= hi))
    recs = [by_step[s] for s in steps]
    if include_final:
        recs += [r for r in finals.values() if (lo is None or r["step"] >= lo) and (hi is None or r["step"] <= hi)]
    gaps = []
    if steps:
        want = set(range(lo if lo is not None else steps[0], (hi if hi is not None else steps[-1]) + 1))
        gaps = sorted(want - set(steps))
    if info is not None:
        info.update(headers=headers, duplicates_replaced=n_dup, bad_lines=n_bad, missing_steps=gaps)
    if n_dup:
        print(f"[warn] {n_dup} duplicate step record(s) (relaunch/resume into the same file): kept the last of each",
              file=sys.stderr)
    if gaps:
        print(f"[warn] {len(gaps)} step(s) missing in range: {gaps[:10]}{' ...' if len(gaps) > 10 else ''}",
              file=sys.stderr)
    return recs


def div(a, b):
    return a / b if b else None


def fmt(x, nd=3, pct=False):
    if x is None:
        return "-"
    if pct:
        return f"{100 * x:.1f}%"
    if isinstance(x, float) and not x.is_integer():
        return f"{x:.{nd}f}"
    return str(int(x))


def fine_percentile(fine, q, fine_bin, cap):
    n = sum(fine)
    if not n:
        return None
    k = max(1, math.ceil(q * n))
    c = 0
    for i, v in enumerate(fine):
        c += v
        if c >= k:
            return cap if i == len(fine) - 1 else i * fine_bin + fine_bin // 2
    return cap


def _num(x):
    return 0 if x is None else x


def aggregate(recs):
    meta = {k: recs[-1].get(k) for k in ("cap", "edges", "fine_bin", "num_generations", "max_staleness")}
    names = list(recs[0]["buckets"])
    out_b = {}
    for key, _ in BUCKETINGS:
        if not all(key in r for r in recs):
            continue
        bn = list(recs[0][key])
        b = {n: dict.fromkeys(GROUP_KEYS, 0) for n in bn}
        for r in recs:
            for n in bn:
                for k in GROUP_KEYS:
                    b[n][k] += _num(r[key][n].get(k, 0))
        out_b[key] = b
    tot = dict.fromkeys(GROUP_KEYS, 0)
    for n in names:
        for k in GROUP_KEYS:
            tot[k] += out_b["buckets"][n][k]
    lens = {}
    for kind in ("generated", "trained", "dropped"):
        n_ = sum(r["len"][kind]["n"] for r in recs)
        s_ = sum(r["len"][kind]["sum"] for r in recs)
        mx = [r["len"][kind]["max"] for r in recs if r["len"][kind]["max"] is not None]
        hist = {n: sum(r["len"][kind]["hist"][n] for r in recs) for n in names}
        fine = [sum(col) for col in zip(*(r["len"][kind]["fine"] for r in recs))]
        top = max(mx) if mx else None
        p50, p90 = (fine_percentile(fine, q, meta["fine_bin"], meta["cap"]) for q in (0.5, 0.9))
        lens[kind] = {"n": n_, "mean": div(s_, n_), "max": top, "hist": hist,
                      "p50": None if p50 is None else min(p50, top), "p90": None if p90 is None else min(p90, top)}
    rw = {k: sum(_num(r["reward"].get(k, 0)) for r in recs) for k in REWARD_KEYS}
    samples = {k: sum(r["samples"].get(k, 0) for r in recs) for k in SAMPLE_KEYS}
    anomalies = {}
    for r in recs:
        for k, v in r["anomalies"].items():
            anomalies[k] = anomalies.get(k, 0) + v
    return {"names": names, "meta": meta, "buckets": out_b["buckets"], "bucketings": out_b, "total": tot,
            "len": lens, "reward": rw, "samples": samples, "anomalies": anomalies,
            "steps": [recs[0]["step"], recs[-1]["step"]], "n_records": len(recs),
            "backlog_last": recs[-1]["cum"]["backlog"], "backlog_max": max(r["cum"]["backlog"] for r in recs),
            "jsonl_failures_last": recs[-1].get("jsonl_failures", 0),
            "hook_ms_mean": div(sum(r["hook_ms"] for r in recs), len(recs))}


def bucket_row(name, a):
    return [name, a["groups"], a["kept_full"], a["dropped_full"], a["partial"], a["short"], a["samples"],
            a["samples_dropped"],
            fmt(div(a["samples_dropped"], a["samples"]), pct=True),
            fmt(div(a["dropped_full"], a["groups"]), pct=True),
            fmt(div(a["partial"], a["groups"]), pct=True),
            fmt(div(a["lag_sum"], a["lag_n"]), 2),
            fmt(div(a["r_kept_full_sum"], a["r_kept_full_n"])),
            fmt(div(a["r_dropped_full_sum"], a["r_dropped_full_n"])),
            fmt(div(a["r_partial_kept_sum"] + a["r_partial_dropped_sum"],
                    a["r_partial_kept_n"] + a["r_partial_dropped_n"]))]


def table(header, rows):
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    out = ["  ".join(h.rjust(w[i]) if i else h.ljust(w[i]) for i, h in enumerate(header))]
    out.append("  ".join("-" * x for x in w))
    for r in rows:
        out.append("  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r)))
    return "\n".join(out)


GROUP_HEADER = ["bucket", "groups", "kept", "dropped", "partial", "short", "samples", "s_drop", "drop%",
                "grp_drop%", "partial%", "lag", "R_trained", "R_dropped", "R_partial"]


def report(agg):
    m, names = agg["meta"], agg["names"]
    s = agg["samples"]
    lines = [f"drop_audit: steps {agg['steps'][0]}..{agg['steps'][1]} ({agg['n_records']} records)  "
             f"ngen={m['num_generations']} max_staleness={m['max_staleness']} cap={m['cap']} edges={m['edges']}",
             f"samples: arrived {s['arrived']}  trained {s['trained']}  kept {s['kept']}  dropped {s['dropped']} "
             f"({fmt(div(s['dropped'], s['arrived']), pct=True)})  backlog(cum kept-collated) last "
             f"{agg['backlog_last']} max {agg['backlog_max']}  jsonl_failures {agg['jsonl_failures_last']}  "
             f"hook {fmt(agg['hook_ms_mean'], 2)} ms/step"]
    for key, title in BUCKETINGS:
        if key not in agg["bucketings"]:
            continue
        b = agg["bucketings"][key]
        lines += ["", title + "; counted in the step that was being built when the group completed",
                  table(GROUP_HEADER, [bucket_row(n, b[n]) for n in b] + [bucket_row("ALL", agg["total"])])]
    lines += ["", "Completion length per SAMPLE (generated = trained + dropped; p50/p90 at fine-bin resolution)"]
    rows = []
    for kind in ("generated", "trained", "dropped"):
        L = agg["len"][kind]
        rows.append([kind, L["n"], fmt(L["mean"], 1), fmt(L["p50"]), fmt(L["p90"]), fmt(L["max"])]
                    + [f"{L['hist'][n]} ({fmt(div(L['hist'][n], L['n']), pct=True)})" for n in names])
    lines.append(table(["set", "n", "mean", "p50", "p90", "max"] + names, rows))
    rw, t = agg["reward"], agg["total"]
    lines += ["",
              "Mean reward: trained samples {}  dropped samples {}  |  fully-kept groups {}  fully-dropped groups {}  "
              "partial groups {}".format(
                  fmt(div(rw["trained_samples_sum"], rw["trained_samples_n"])),
                  fmt(div(rw["dropped_samples_sum"], rw["dropped_samples_n"])),
                  fmt(div(t["r_kept_full_sum"], t["r_kept_full_n"])),
                  fmt(div(t["r_dropped_full_sum"], t["r_dropped_full_n"])),
                  fmt(div(t["r_partial_kept_sum"] + t["r_partial_dropped_sum"],
                          t["r_partial_kept_n"] + t["r_partial_dropped_n"]))),
              "anomalies: " + (", ".join(f"{k}={v}" for k, v in sorted(agg["anomalies"].items())) or "none")
              + f"  (total {sum(agg['anomalies'].values())})"]
    return "\n".join(lines)


def per_step(recs):
    names = list(recs[0]["buckets"])
    rows = []
    for r in recs:
        rows.append([r["step"], r["model_version"], r["samples"]["arrived"], r["samples"].get("trained", "-"),
                     r["samples"]["dropped"], fmt(r["samples"]["drop_frac"], pct=True), r["groups"]["groups"],
                     r["groups"]["dropped_full"], r["groups"]["partial"]]
                    + [fmt(r["buckets"][n]["drop_frac"], pct=True) for n in names]
                    + [fmt(r["len"]["generated"]["mean"], 0), fmt(r["len"]["trained"]["mean"], 0),
                       r["cum"]["backlog"], sum(r["anomalies"].values())])
    return table(["step", "ver", "arrived", "trained", "dropped", "drop%", "groups", "g_drop", "g_part"]
                 + [f"{n}%" for n in names] + ["len_gen", "len_trn", "backlog", "anom"], rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--from", dest="lo", type=int, default=None, help="first step (inclusive)")
    ap.add_argument("--to", dest="hi", type=int, default=None, help="last step (inclusive)")
    ap.add_argument("--per-step", action="store_true", help="also print one row per step")
    ap.add_argument("--include-final", action="store_true",
                    help="include the on_train_end records (pulled/collated after the last step, never trained)")
    ap.add_argument("--json", action="store_true", help="print the aggregate as JSON instead of tables")
    a = ap.parse_args(argv)
    info = {}
    recs = load(a.path, a.lo, a.hi, a.include_final, info)
    if not recs:
        print("no records in range", file=sys.stderr)
        return 1
    agg = aggregate(recs)
    agg["file"] = {"segments": [{k: h.get(k) for k in ("seg", "host", "pid", "start_time", "resume_step")}
                                for h in info["headers"]],
                   "duplicates_replaced": info["duplicates_replaced"], "bad_lines": info["bad_lines"],
                   "missing_steps": info["missing_steps"]}
    if a.json:
        print(json.dumps(agg, indent=1))
        return 0
    print(report(agg))
    f = agg["file"]
    print(f"\nfile: {len(f['segments'])} segment(s) {[s['resume_step'] for s in f['segments']]} (resume steps), "
          f"{f['duplicates_replaced']} duplicate step record(s) replaced, {f['bad_lines']} bad line(s), "
          f"{len(f['missing_steps'])} missing step(s)")
    if a.per_step:
        print()
        print(per_step(recs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
