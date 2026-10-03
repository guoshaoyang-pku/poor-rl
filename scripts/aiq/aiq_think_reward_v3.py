"""AIQ reward for the thinking-template RL line (ckpt-57 start, G32, 16k cap). 2026-10-03.

New module; the legacy `ranking_reward.py` (concordance reward) is left untouched.

Reward (defaults follow the 2026-10-03 task brief; all four constants are env-overridable):
  ranking : exact order = +1; otherwise inv >= 1 -> 0.5 / inv
            (inv = Kendall inversions vs gold, 0..10; inv=1 -> 0.5, inv=10 -> 0.0)
  select  : correct letter = +1, wrong = 0
  unparsed: -0.5   (AIQ_R_UNPARSED)
  truncated (completion hit the token cap): -2.0   (AIQ_R_TRUNC)

Answer extraction (thinking template): only text AFTER the last `</think>` counts, and
inside it only the LAST `<answer>...</answer>` tag. With TRL's async worker the completion
arrives as a parsed assistant message whose `content` is already the post-think part
(`reasoning_content` holds the thought; when `</think>` never appears, `content` is
empty) -- so an `<answer>` written inside the thinking block never scores.

Optional hallucination hook (master switch AIQ_HALLUC=1; OFF by default):
  * sampling: every pseudo-step of AIQ_HALLUC_GROUPS_PER_STEP groups (default 32 = Q per
    step) K groups are drawn at random (AIQ_HALLUC_K, default 2, 1..groups_per_step; a live
    override is read from AIQ_HALLUC_K_FILE each pseudo-step; cumulative cost at
    AIQ_HALLUC_PRICE_PER_M yuan / 1M input tokens, above AIQ_HALLUC_BUDGET_YUAN=250 K drops
    to AIQ_HALLUC_K_OVER_BUDGET=2); ALL NGEN
    rollouts of a drawn group go to the Luna judge (never split a group). The reward is
    called once per group, before the group is handed to the trainer, so the judge result
    is always in place before the group trains.
  * judge calls are issued concurrently from a persistent thread pool; the group waits at
    most AIQ_HALLUC_TIMEOUT_S (90 s) wall clock. Late / failed calls count as "no penalty"
    and are logged; abandoned calls finish in the background and are discarded.
  * judge provider `cctq` from eval_keys.json (AIQ_EVAL_KEYS), model gpt-5.6-luna; two
    fields: hallucination none|minor|severe, analysis real|restate_only (uncertain -> real).
  * penalty tiers (2026-10-03 v2 calibration, halluc/report_v2.md):
      Luna severe, kind not_read | gibberish                   -> -2 (AIQ_R_HALLUC)
      Luna severe, kind fabricated_basis (or no kind)          -> log only
      Luna analysis restate_only and exact                     -> x0.5 (AIQ_R_RESTATE_MULT)
      rule detector severe (every rollout while the hook is on) -> log only
          (AIQ_HALLUC_RULES_REWARD=1 would make it -2; precision 73% / recall 9%)
    AIQ_HALLUC_LUNA_REWARD=0 turns the Luna tiers into log-only.
    Penalise-only; truncated rollouts (-2 already) are not judged.
  * offline nodes: AIQ_HALLUC_BASE_URL=http://127.0.0.1:3129/v1 without AIQ_EVAL_KEYS uses
    judge_relay.py (runs on a machine with egress, injects the key; reverse SSH tunnel), so
    no credential lives on the cluster. AIQ_HALLUC_PROXY (CONNECT proxy) is an alternative.
  * per pseudo-step log record {"halluc_step": ...}: judge calls, timeouts, failures,
    rule_severe, luna_severe and per kind (not_read / gibberish / fabricated_basis),
    luna_restate_only, and how many penalties were applied to reward.
The API key is read at call time and is never logged or stored.

Per-group log line (RLFORGE_TASK_LOG): source / task / exact / inv list / zero-variance
flag, plus the legacy fields (n_mcq, n_rank, by_source, ...) so old parsers keep working.

v3 (2026-10-03, non-blocking scoring; same reward values). Shipped as a NEW module,
aiq_think_reward_v3.py (REWARD=aiq_think_reward_v3:think_reward), so runs that import the shared
aiq_think_reward.py keep the stock judge order and log format:
  * thread-safe under rlforge's concurrent score loop (several think_reward calls at once):
    lazy judge client / pool under a lock; per-group counters merged into the pseudo-step under
    the lock and the step logged only once ALL of its groups are done (stock logged it when the
    next step's first group arrived -> counts of groups still waiting on the judge were lost);
    rule-detector cache is a locked dict (stock's 1-entry cache could raise KeyError under
    concurrency, silently skipping the whole halluc hook incl. the judge); log appends locked
    (16k-token sample rows are several write() calls).
  * judge pool threads are daemon threads (a late judge call never blocks process exit).
  * judge calls are submitted in random order (AIQ_HALLUC_SHUFFLE=1, default): stock submitted
    in completion order = shortest rollouts first, so with ~70% timeouts the verdicts that did
    arrive came from the shortest rollouts of each group.
  * rlforge integration (only when rlforge passes the kwargs): rlforge_info (group_id in,
    judged out -> per-sample metric rlforge/judged, used for --judged-max-staleness) and the
    early hook rlforge_on_rollout (judge call submitted as soon as each rollout finishes).
    With a group_id, selection is by dispatch order (pseudo-step = group_id // groups_per_step);
    AIQ_HALLUC_SELECT=scored restores stock scoring-order selection.
  * admission control (off by default): AIQ_HALLUC_RPM (token bucket, burst AIQ_HALLUC_BURST,
    default RPM/4) and AIQ_HALLUC_JUDGE_FRAC (Bernoulli subsample of a selected group's live
    rollouts). Calls refused by either are not sent (skipped_rate / skipped_frac in the step
    record), instead of queueing behind the relay's ~10 req/min upstream limit and timing out.
    JUDGE_FRAC < 1 deliberately SPLITS a selected group (only ~FRAC of its rollouts get a
    verdict), i.e. it gives up the "never split a group" rule above in exchange for judging more
    prompts per call budget. The RPM bucket refuses calls immediately (no waiting): its burst
    (AIQ_HALLUC_BURST) must be >= the calls one judged group sends (~NGEN x FRAC), or it, not
    FRAC, caps every group at `burst` verdicts.
  * AIQ_HALLUC_TAIL_S (off by default): once a judged group is complete, wait at most this long
    for its outstanding judge calls (with early calls most verdicts are already in by then);
    bounds the judged group's extra delay = its extra staleness.
  * new step-record fields: early_calls, skipped_rate, skipped_frac, late_calls,
    judge_queue_mean_s / judge_queue_max_s (time a call waited for a judge thread), and the
    length-bias monitor live_by_len_q / sent_by_len_q / verdict_by_len_q: for the judged groups'
    non-truncated rollouts, counts per within-group completion-length quartile (short -> long) of
    rollouts, calls sent and verdicts received in time. verdict/live should be flat across the
    four quartiles; a falling profile means long CoTs are judged less (early hooks under
    saturation do this).
  * record order: a pseudo-step record is written when the LAST of its groups closes, so records
    can appear out of pseudo_step order and the cum_* fields (cumulative at write time) are not
    monotonic in pseudo_step. Sort by pseudo_step (or t) when analysing. A judge call that finishes
    after its group stopped waiting is counted in its OWN pseudo-step (late_calls, latency) if
    that step is still open, else logged as {"halluc_late": {"pseudo_step", "latency_s",
    "prompt_tokens"}} (stock pooled late calls into whichever step was logged next).
  * early hooks (rlforge --reward-early-hooks, EXPERIMENTAL): calls are submitted in completion
    order, so under judge saturation (timeouts / RPM refusals) the short rollouts of each group
    get the verdicts and the long ones time out -- the bias the shuffle removes. Keep it off unless
    judge capacity clearly exceeds demand (timeout_rate ~0, skipped_rate 0); watch verdict_by_len_q.
"""
from __future__ import annotations

import functools
import json
import os
import random
import re
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
THINK_CLOSE = "</think>"
RANK_BODY_RE = re.compile(r"^[A-Ea-e](?:\s*[<>]\s*[A-Ea-e]){4}$")
LETTER_BODY_RE = re.compile(r"^[\s(\[*]*([A-Ea-e])[\s)\].:*]*$")
SEP_RE = re.compile(r"\s*[<>]\s*")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


R_EXACT = 1.0
R_UNPARSED = _env_float("AIQ_R_UNPARSED", -0.5)
R_TRUNC = _env_float("AIQ_R_TRUNC", -2.0)
R_HALLUC = _env_float("AIQ_R_HALLUC", -2.0)
R_RESTATE_MULT = _env_float("AIQ_R_RESTATE_MULT", 0.5)


# ----------------------------------------------------------------------------- parsing
def after_think_text(completion):
    """Text after the last </think>; '' when the thinking block never closed.

    Accepts a raw string (eval / probe path) or TRL's parsed message list.
    """
    if isinstance(completion, list):
        msgs = [m for m in completion if isinstance(m, dict) and m.get("role", "assistant") == "assistant"]
        if not msgs:
            return ""
        text = msgs[-1].get("content") or ""
        if not isinstance(text, str):
            text = str(text)
        # defensive: a template that did not split reasoning leaves </think> in content
        if THINK_CLOSE in text:
            text = text.rsplit(THINK_CLOSE, 1)[1]
        return text
    text = completion if isinstance(completion, str) else str(completion)
    if THINK_CLOSE not in text:
        return ""
    return text.rsplit(THINK_CLOSE, 1)[1]


def thinking_text(completion) -> str:
    """Visible reasoning (for the judge)."""
    if isinstance(completion, list):
        msgs = [m for m in completion if isinstance(m, dict)]
        if not msgs:
            return ""
        m = msgs[-1]
        r = m.get("reasoning_content") or m.get("reasoning") or m.get("thinking") or ""
        c = m.get("content") or ""
        if not r and THINK_CLOSE in c:
            r = c.rsplit(THINK_CLOSE, 1)[0]
        return str(r)
    text = completion if isinstance(completion, str) else str(completion)
    return text.rsplit(THINK_CLOSE, 1)[0] if THINK_CLOSE in text else text


def final_answer_body(completion):
    tags = ANSWER_TAG_RE.findall(after_think_text(completion))
    return tags[-1].strip() if tags else None


def parse_ranking(body):
    """'E<A<D<B<C' (or with '>' / spaces / lowercase) -> ('E','A','D','B','C'), else None.
    The letter sequence is read in written order; the separator is not interpreted."""
    if body is None or not RANK_BODY_RE.match(body.strip()):
        return None
    letters = tuple(p.upper() for p in SEP_RE.split(body.strip()) if p)
    return letters if len(letters) == 5 and set(letters) == set("ABCDE") else None


def parse_letter(body, num_choices=None):
    if body is None:
        return None
    m = LETTER_BODY_RE.match(body)
    if not m:
        return None
    letter = m.group(1).upper()
    if num_choices:
        try:
            if ord(letter) - ord("A") >= int(num_choices):
                return None
        except (TypeError, ValueError):
            pass
    return letter


def gold_letters(gold):
    return tuple(p.upper() for p in SEP_RE.split(str(gold).strip()) if p)


def is_ranking_gold(gold) -> bool:
    return len(gold_letters(gold)) > 1


def inversions(pred, gold) -> int:
    pos = {c: i for i, c in enumerate(gold)}
    seq = [pos[c] for c in pred]
    return sum(seq[i] > seq[j] for i in range(len(seq)) for j in range(i + 1, len(seq)))


def ranking_value(inv: int) -> float:
    if inv == 0:
        return R_EXACT
    return 0.5 / inv


def score_one(completion, gold, n_tokens=None, cap=16384, num_choices=None):
    """-> dict(reward, status in {exact, wrong, unparsed, trunc}, inv or None, pred)."""
    ranking = is_ranking_gold(gold)
    if n_tokens is not None and n_tokens >= cap:
        return {"reward": R_TRUNC, "status": "trunc", "inv": None, "pred": None, "ranking": ranking}
    body = final_answer_body(completion)
    if ranking:
        g = gold_letters(gold)
        if len(g) != 5 or set(g) != set("ABCDE"):
            raise ValueError(f"invalid gold ranking {gold!r}")
        pred = parse_ranking(body)
        if pred is None:
            return {"reward": R_UNPARSED, "status": "unparsed", "inv": None, "pred": None, "ranking": True}
        inv = inversions(pred, g)
        return {"reward": ranking_value(inv), "status": "exact" if inv == 0 else "wrong", "inv": inv,
                "pred": "<".join(pred), "ranking": True}
    pred = parse_letter(body, num_choices)
    if pred is None:
        return {"reward": R_UNPARSED, "status": "unparsed", "inv": None, "pred": None, "ranking": False}
    ok = pred == str(gold).strip().upper()
    return {"reward": R_EXACT if ok else 0.0, "status": "exact" if ok else "wrong", "inv": None,
            "pred": pred, "ranking": False}


# ------------------------------------------------------------------ hallucination hook
# Judge prompt: lenient-v2 with severe_kind, copied verbatim from
# aiq_rl/tools/reward_design/halluc_judge_cctq.py (calibrated 2026-10-03, report_v2.md).
JUDGE_SYS = "You are a lenient but careful auditor of reasoning traces. Output only one JSON object."
JUDGE_TMPL = """Below is a benchmark question, a model's visible chain-of-thought (CoT), and its final answer.
Judge the CoT's faithfulness to the question. Do NOT judge answer correctness. Be LENIENT: the "severe" label drives a heavy penalty, so reserve it for clear cases.

hallucination labels:
- "none": the model read the question and engages with THIS question's specifics. Small slips are fine. General ML knowledge and correct arithmetic from the prompt are fine.
- "minor": tolerable problems: a few invented details or wrong facts, hallucinations inside intermediate reasoning that are not the main basis of the conclusion, some vague jargon, mild repetition.
- "severe": ONLY one of these three (set severe_kind accordingly):
  "not_read": did not read the question - generic/template boilerplate that cites NONE of this question's specific parameters/configurations;
  "fabricated_basis": a fabricated or contradicted question fact (parameter, value, component, choice attribution) is the MAIN basis of the conclusion;
  "gibberish": large stretches of self-invented terms/gibberish or degenerate repetition.

analysis labels:
- "restate_only": the CoT only restates/paraphrases the question (lists the choices/configs) and then gives an answer, WITHOUT any comparison or reasoning about why one option is better/worse.
- "real": the CoT contains at least some comparison or reasoning across options. If unsure, choose "real".

reason: one short sentence (<=30 words) justifying both labels.
evidence_quote: shortest verbatim quote from the CoT supporting the hallucination label ("" if none).

Output exactly: {{"hallucination": "none|minor|severe", "severe_kind": "not_read|fabricated_basis|gibberish|", "analysis": "real|restate_only", "reason": "...", "evidence_quote": "..."}}

===== QUESTION =====
{question}
===== MODEL COT =====
{cot}
===== FINAL ANSWER =====
{answer}
"""

_HSTATE = {"groups": 0, "picked": set(), "client": None, "pool": None,
           "lock": threading.Lock(), "step": None, "prompt_tokens": 0, "calls": 0,
           # v3 (concurrency-safe accounting):
           "steps": {},     # pseudo_step -> {"st": stats, "picks": set, "open": n, "closed": n}
           "bucket": None}  # AIQ_HALLUC_RPM token bucket
_INIT_LOCK = threading.Lock()  # lazy client / pool / bucket construction
_LOG_LOCK = threading.Lock()   # one writer at a time on the task / sample logs
_SKIP_RATE, _SKIP_FRAC = "skip_rate", "skip_frac"  # early-hook markers: call deliberately not sent


class _DaemonPool(ThreadPoolExecutor):
    """ThreadPoolExecutor with daemon workers that are not joined at interpreter exit, so a
    judge call still in flight (relay queue + 90 s client timeout) never holds up shutdown.
    CPython 3.10-3.12 private layout; anything else falls back to stock workers."""

    def _adjust_thread_count(self):
        from concurrent.futures import thread as _cft
        if not hasattr(_cft, "_worker") or sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
            return super()._adjust_thread_count()
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        n = len(self._threads)
        if n < self._max_workers:
            t = threading.Thread(name=f"{self._thread_name_prefix}_{n}", target=_cft._worker, daemon=True,
                                 args=(weakref.ref(self, weakref_cb), self._work_queue,
                                       self._initializer, self._initargs))
            t.start()
            self._threads.add(t)


def _halluc_enabled() -> bool:
    return os.environ.get("AIQ_HALLUC", "0") == "1"


def _halluc_cost_yuan() -> float:
    return _HSTATE["prompt_tokens"] * _env_float("AIQ_HALLUC_PRICE_PER_M", 0.1) / 1e6


def _halluc_k() -> int:
    """Groups judged per pseudo-step. AIQ_HALLUC_K_FILE (if it exists and holds an int) overrides
    AIQ_HALLUC_K so K can be changed live; once the cumulative judge cost passes
    AIQ_HALLUC_BUDGET_YUAN (250) K drops to AIQ_HALLUC_K_OVER_BUDGET (2)."""
    k = int(os.environ.get("AIQ_HALLUC_K", "2"))
    kf = os.environ.get("AIQ_HALLUC_K_FILE", "")
    if kf:
        try:
            k = int(open(kf).read().strip())
        except (OSError, ValueError):
            pass
    if _halluc_cost_yuan() > _env_float("AIQ_HALLUC_BUDGET_YUAN", 250.0):
        k = min(k, int(os.environ.get("AIQ_HALLUC_K_OVER_BUDGET", "2")))
    return min(_groups_per_step(), max(1, k))


def _groups_per_step() -> int:
    return max(1, int(os.environ.get("AIQ_HALLUC_GROUPS_PER_STEP", "32")))


_COUNTERS = ("judged_groups", "judge_calls", "timeouts", "fails", "rule_severe", "luna_severe",
             "luna_not_read", "luna_gibberish", "luna_fabricated_basis", "luna_severe_nokind",
             "luna_restate_only", "applied_rule", "applied_luna_severe", "applied_restate",
             "early_calls", "skipped_rate", "skipped_frac")
# length-bias monitor: per within-group completion-length quartile (short -> long) of a judged
# group's live rollouts: rollouts / calls sent / verdicts received in time
_QFIELDS = ("live_by_len_q", "sent_by_len_q", "verdict_by_len_q")


def _new_step(idx: int) -> dict:
    st = {"pseudo_step": idx, "groups": 0}
    st.update({k: 0 for k in _COUNTERS})
    st.update({"late_calls": 0, "k": None, "latencies": [], "queue_waits": [], "prompt_tokens": 0})
    st.update({k: [0, 0, 0, 0] for k in _QFIELDS})
    return st


def _local_stats() -> dict:
    d = {k: 0 for k in _COUNTERS}
    d.update({"latencies": [], "queue_waits": [], "prompt_tokens": 0})
    d.update({k: [0, 0, 0, 0] for k in _QFIELDS})
    return d


def _step_rec(step_idx: int) -> dict:
    """Caller holds _HSTATE['lock']. Creates the pseudo-step record and draws its K picks on
    first touch (stock drew them when the step's first group arrived: same thing)."""
    rec = _HSTATE["steps"].get(step_idx)
    if rec is None:
        gps = _groups_per_step()
        st = _new_step(step_idx)
        k = _halluc_k()
        st["k"] = k
        rec = {"st": st, "picks": set(random.sample(range(gps), min(k, gps))), "open": 0, "closed": 0}
        _HSTATE["steps"][step_idx] = rec
        _HSTATE["step"], _HSTATE["picked"] = st, rec["picks"]  # legacy pointers (newest step)
    return rec


def _halluc_slot(group_id=None):
    """-> (selected?, pseudo_step). Opens the group in its pseudo-step's accounting.
    group_id None: stock semantics, the g-th group *scored* (pseudo-step = g // groups_per_step).
    group_id given (rlforge passes it): the g-th group *dispatched*; order-independent, which is
    what concurrent scoring and the early hook need."""
    gps = _groups_per_step()
    with _HSTATE["lock"]:
        if group_id is None:
            g = _HSTATE["groups"]
            _HSTATE["groups"] += 1
        else:
            g = int(group_id)
        pos, step_idx = g % gps, g // gps
        rec = _step_rec(step_idx)
        rec["open"] += 1
        rec["st"]["groups"] += 1
        return pos in rec["picks"], step_idx


def _is_selected_gid(group_id) -> bool:
    gps = _groups_per_step()
    with _HSTATE["lock"]:
        return (int(group_id) % gps) in _step_rec(int(group_id) // gps)["picks"]


def _close_group(step_idx: int, local: dict) -> None:
    """Merge one group's counters into its pseudo-step; log the step once all of its groups are
    done (stock logged it when the next step's first group arrived, which under concurrent
    scoring would drop the counts of groups still waiting on the judge)."""
    gps = _groups_per_step()
    out = None
    with _HSTATE["lock"]:
        rec = _HSTATE["steps"].get(step_idx)
        if rec is None:
            return
        st = rec["st"]
        for k in _COUNTERS:
            st[k] += local[k]
        st["latencies"].extend(local["latencies"])
        st["queue_waits"].extend(local["queue_waits"])
        st["prompt_tokens"] += local["prompt_tokens"]
        for k in _QFIELDS:
            st[k] = [a + b for a, b in zip(st[k], local[k])]
        rec["open"] -= 1
        rec["closed"] += 1
        if rec["closed"] >= gps and rec["open"] <= 0:
            del _HSTATE["steps"][step_idx]
            out = _summarize_step(st)
    if out is not None:
        _log({"t": round(time.time(), 1), "halluc_step": out})


def _summarize_step(st: dict) -> dict:
    out = {k: v for k, v in st.items() if k not in ("latencies", "queue_waits")}
    lat = sorted(st["latencies"])
    if lat:
        out["latency_mean_s"] = round(sum(lat) / len(lat), 2)
        out["latency_p90_s"] = round(lat[int(0.9 * (len(lat) - 1))], 2)
        out["latency_max_s"] = round(lat[-1], 2)
    qw = st.get("queue_waits") or []
    if qw:
        out["judge_queue_mean_s"] = round(sum(qw) / len(qw), 2)
        out["judge_queue_max_s"] = round(max(qw), 2)
    out["timeout_rate"] = round(st["timeouts"] / st["judge_calls"], 4) if st["judge_calls"] else None
    out["cum_calls"] = _HSTATE["calls"]
    out["cum_prompt_tokens"] = _HSTATE["prompt_tokens"]
    out["cum_cost_yuan"] = round(_halluc_cost_yuan(), 4)
    return out


def _judge_client():
    if _HSTATE["client"] is None:
        with _INIT_LOCK:
            if _HSTATE["client"] is None:
                from openai import OpenAI  # imported lazily: not needed when the hook is off
                path = os.environ.get("AIQ_EVAL_KEYS", "")
                provider = os.environ.get("AIQ_HALLUC_PROVIDER", "cctq")
                if path:
                    with open(path) as fh:
                        k = json.load(fh)[provider]
                else:  # key-injecting relay (judge_relay.py): no key on this machine
                    k = {"base_url": "", "api_key": "relay-injects-key"}
                base_url = os.environ.get("AIQ_HALLUC_BASE_URL") or k["base_url"]
                kw = {}
                proxy = os.environ.get("AIQ_HALLUC_PROXY", "")  # e.g. http://127.0.0.1:3128 (offline nodes)
                if proxy:
                    import httpx
                    kw["http_client"] = httpx.Client(proxy=proxy, timeout=float(os.environ.get("AIQ_HALLUC_TIMEOUT_S", "90")))
                _HSTATE["client"] = OpenAI(base_url=base_url, api_key=k["api_key"], max_retries=0,
                                           timeout=float(os.environ.get("AIQ_HALLUC_TIMEOUT_S", "90")), **kw)
    return _HSTATE["client"]


def _pool():
    if _HSTATE["pool"] is None:
        with _INIT_LOCK:
            if _HSTATE["pool"] is None:
                _HSTATE["pool"] = _DaemonPool(int(os.environ.get("AIQ_HALLUC_THREADS", "64")),
                                              thread_name_prefix="aiq-judge")
    return _HSTATE["pool"]


class _Bucket:
    """Token bucket for AIQ_HALLUC_RPM: a call that cannot get a token is not sent at all
    (counted skipped_rate) instead of queueing in the relay and timing out."""

    def __init__(self, rpm: float, burst: float):
        self.rpm, self.rate, self.cap = rpm, rpm / 60.0, max(1.0, burst)
        self.tokens, self.t = self.cap, time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


def _admit() -> bool:
    rpm = _env_float("AIQ_HALLUC_RPM", 0.0)
    if rpm <= 0:
        return True
    b = _HSTATE["bucket"]
    if b is None or b.rpm != rpm:
        with _INIT_LOCK:
            b = _HSTATE["bucket"]
            if b is None or b.rpm != rpm:
                b = _HSTATE["bucket"] = _Bucket(rpm, _env_float("AIQ_HALLUC_BURST", max(1.0, rpm / 4.0)))
    return b.take()


def _frac_ok() -> bool:
    frac = _env_float("AIQ_HALLUC_JUDGE_FRAC", 1.0)
    return frac >= 1.0 or random.random() < frac


def _prompt_text(prompt) -> str:
    if isinstance(prompt, list):
        return "\n".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict) and m.get("role") == "user")
    return str(prompt)


def _judge_one_timed(question: str, cot: str, answer: str):
    """-> (verdict or None, latency_s or None, prompt_tokens). verdict as judge_one()."""
    if len(cot) > 40000:
        cot = cot[:20000] + "\n...[truncated]...\n" + cot[-20000:]
    msg = JUDGE_TMPL.format(question=question, cot=cot or "(empty)", answer=answer or "(none)")
    model = os.environ.get("AIQ_HALLUC_MODEL", "gpt-5.6-luna")
    try:
        t0 = time.time()
        resp = _judge_client().chat.completions.create(
            model=model, temperature=0, max_tokens=800,
            messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": msg}])
        lat = time.time() - t0
        u = getattr(resp, "usage", None)
        ptok = int(getattr(u, "prompt_tokens", 0) or 0) if u is not None else 0
        with _HSTATE["lock"]:
            _HSTATE["calls"] += 1
            _HSTATE["prompt_tokens"] += ptok
        txt = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", txt, re.S)
        j = json.loads(m.group(0)) if m else {}
    except Exception as e:  # noqa: BLE001 - log the type only (key-free)
        _log({"t": round(time.time(), 1), "halluc_judge_error": type(e).__name__})
        return None, None, 0
    label = j.get("hallucination")
    if label not in ("none", "minor", "severe"):
        return None, lat, ptok
    kind = j.get("severe_kind") if j.get("severe_kind") in ("not_read", "fabricated_basis", "gibberish") else ""
    return (label, (kind if label == "severe" else ""),
            ("restate_only" if j.get("analysis") == "restate_only" else "real")), lat, ptok


def judge_one(question: str, cot: str, answer: str):
    """-> (hallucination, severe_kind, analysis): hallucination none|minor|severe, severe_kind
    not_read|fabricated_basis|gibberish|'' and analysis real|restate_only; None on judge
    failure (never penalise). Unknown analysis -> real; unknown kind -> ''."""
    return _judge_one_timed(question, cot, answer)[0]


class _JudgeHandle:
    __slots__ = ("future", "t_submit", "t_start")


def _submit_judge(question: str, cot: str, answer: str) -> _JudgeHandle:
    h = _JudgeHandle()
    h.t_submit, h.t_start = time.time(), None

    def run():
        h.t_start = time.time()
        return _judge_one_timed(question, cot, answer)

    h.future = _pool().submit(run)
    return h


def _late_cb(step_idx, fut) -> None:
    """A judge call finished after its group stopped waiting (billed, verdict discarded): count it
    in its own pseudo-step if that is still open, else log it on its own with that pseudo_step."""
    if fut.cancelled():
        return
    try:
        _, lat, ptok = fut.result()
    except Exception:  # noqa: BLE001
        return
    if lat is None:
        return
    with _HSTATE["lock"]:
        rec = _HSTATE["steps"].get(step_idx) if step_idx is not None else None
        if rec is not None:
            rec["st"]["late_calls"] += 1
            rec["st"]["latencies"].append(lat)
            rec["st"]["prompt_tokens"] += ptok
            return
    _log({"t": round(time.time(), 1), "halluc_late": {"pseudo_step": step_idx, "latency_s": round(lat, 2),
                                                       "prompt_tokens": ptok}})


# ---- rule detector (port of tools/reward_design/halluc_judge.py:rule_detect) ----
NUM_RE = re.compile(r"(?<![A-Za-z_])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![A-Za-z_\d])")
IDENT_RE = re.compile(r"\b([A-Za-z]+_[A-Za-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z]+\d+[A-Za-z0-9]*)\b")


def _numset(text):
    out = set()
    for m in NUM_RE.findall(text):
        try:
            out.add(round(float(m), 10))
        except ValueError:
            pass
    return out


_RULE_CACHE: dict = {}
_RULE_LOCK = threading.Lock()


def rule_severe(prompt_text: str, cot: str, _cache=None) -> bool:
    """severe = >=2 distinct parameter-like numbers or >=2 identifiers absent from the prompt
    (numbers derivable by one +,-,*,/ from prompt numbers are allowed).
    v3: the per-prompt cache is a small locked dict (stock used a 1-entry default-arg dict that a
    concurrent caller could clear between the fill and the read -> KeyError -> whole halluc hook
    skipped for that group, including the Luna judge)."""
    key = hash(prompt_text)
    with _RULE_LOCK:
        entry = _RULE_CACHE.get(key)
    if entry is None:
        pn = _numset(prompt_text)
        pl = [x for x in pn if 0 < abs(x) < 1e7][:200]
        der = set()
        for a in pl:
            for b in pl:
                der.update((round(a * b, 10), round(a / b, 10), round(a + b, 10), round(a - b, 10)))
        entry = (pn | der, prompt_text.lower())
        with _RULE_LOCK:
            while len(_RULE_CACHE) >= 64:
                _RULE_CACHE.pop(next(iter(_RULE_CACHE)))
            _RULE_CACHE[key] = entry
    known, plow = entry
    unseen = set()
    for m in NUM_RE.findall(cot):
        try:
            v = round(float(m), 10)
        except ValueError:
            continue
        if ("." in m or "e" in m.lower() or v >= 11) and v not in known:
            unseen.add(m)
    ids = {t for t in IDENT_RE.findall(cot) if t.lower() not in plow}
    return len(unseen) >= 2 or len(ids) >= 2


def _len_quartiles(live, lengths):
    """index -> within-group completion-length quartile 0..3 (short -> long) among `live`."""
    if not lengths:
        return {}
    ranked = sorted(live, key=lambda i: (lengths[i], i))
    n = max(1, len(ranked))
    return {i: min(3, r * 4 // n) for r, i in enumerate(ranked)}


def _apply_halluc(rewards, statuses, completions, prompts, selected, st, step_idx=None, early=None,
                  lengths=None):
    """Mutates rewards in place; accumulates into `st` (this group's private counters, merged
    into the pseudo-step under the lock by _close_group); returns the per-group record.
    early: optional list aligned with completions of _JudgeHandle / skip marker / None, from
    the rlforge early hook (calls already submitted when each rollout finished generating)."""
    from concurrent.futures import wait
    question = _prompt_text(prompts[0]) if prompts else ""
    live = [i for i, s in enumerate(statuses) if s != "trunc"]
    rec = {"selected": bool(selected)}
    rule_on = os.environ.get("AIQ_HALLUC_RULES", "1") == "1"
    rule_reward = os.environ.get("AIQ_HALLUC_RULES_REWARD", "0") == "1"
    luna_reward = os.environ.get("AIQ_HALLUC_LUNA_REWARD", "1") == "1"
    if rule_on:
        flags = [i for i in live if rule_severe(question, thinking_text(completions[i]))]
        rec["rule_severe"] = len(flags)
        st["rule_severe"] += len(flags)
        if rule_reward:  # precision 73% / recall 9% in the v2 calibration -> log-only by default
            for i in flags:
                if rewards[i] > R_HALLUC:
                    rewards[i] = R_HALLUC
                    st["applied_rule"] += 1
    if not selected:
        return rec
    st["judged_groups"] += 1
    t0 = time.time()
    quart = _len_quartiles(live, lengths)
    for i in live:
        if i in quart:
            st["live_by_len_q"][quart[i]] += 1
    order = list(live)
    # v3: submit in random order. Stock submitted in completion order (= shortest rollouts
    # first) into an 8-thread pool that cannot finish a group in 90 s, so verdicts were biased
    # towards the shortest rollouts of every judged group.
    if os.environ.get("AIQ_HALLUC_SHUFFLE", "1") == "1":
        random.shuffle(order)
    handles = {}
    for i in order:
        h = early[i] if early is not None and i < len(early) else None
        if isinstance(h, _JudgeHandle):
            handles[i] = h
            st["early_calls"] += 1
            continue
        if h == _SKIP_RATE:
            st["skipped_rate"] += 1
            continue
        if h == _SKIP_FRAC:
            st["skipped_frac"] += 1
            continue
        if not _frac_ok():
            st["skipped_frac"] += 1
            continue
        if not _admit():
            st["skipped_rate"] += 1
            continue
        handles[i] = _submit_judge(question, thinking_text(completions[i]), final_answer_body(completions[i]) or "")
    st["judge_calls"] += len(handles)
    for i in handles:
        if i in quart:
            st["sent_by_len_q"][quart[i]] += 1
    timeout_s = _env_float("AIQ_HALLUC_TIMEOUT_S", 90.0)
    done, pending = set(), set()
    if handles:
        # every call gets its full budget from its own submission; early calls are older
        deadline = max(h.t_submit for h in handles.values()) + timeout_s
        tail_s = _env_float("AIQ_HALLUC_TAIL_S", 0.0)
        if tail_s > 0:  # cap the judged group's extra delay once it is complete (bounds its staleness)
            deadline = min(deadline, time.time() + tail_s)
        done, pending = wait([h.future for h in handles.values()], timeout=max(0.0, deadline - time.time()))
    for f in pending:
        if not f.cancel():  # running: finishes in the background, latency logged as late
            f.add_done_callback(functools.partial(_late_cb, step_idx))
    c = {"none": 0, "minor": 0, "severe": 0, "not_read": 0, "gibberish": 0, "fabricated_basis": 0,
         "severe_nokind": 0, "restate_only": 0, "timeout": len(pending), "fail": 0}
    for i, h in handles.items():
        f = h.future
        if f not in done:
            continue
        try:
            res, lat, ptok = f.result()
        except Exception:  # noqa: BLE001
            res, lat, ptok = None, None, 0
        if lat is not None:
            st["latencies"].append(lat)
            st["prompt_tokens"] += ptok
        if h.t_start is not None:
            st["queue_waits"].append(h.t_start - h.t_submit)
        if res is None:
            c["fail"] += 1
            continue
        if i in quart:
            st["verdict_by_len_q"][quart[i]] += 1
        lab, kind, analysis = res
        restate = analysis == "restate_only"
        c[lab] += 1
        c["restate_only"] += restate
        st["luna_restate_only"] += restate
        if lab == "severe":
            st["luna_severe"] += 1
            key = kind or "severe_nokind"
            c[key] += 1
            st["luna_" + key] += 1
        if not luna_reward:
            continue
        # penalised severe kinds: not_read / gibberish only. fabricated_basis (17-37% at the SFT
        # start) and kind-less severe stay log-only. restate_only halves exact answers only.
        if lab == "severe" and kind in ("not_read", "gibberish"):
            if rewards[i] > R_HALLUC:
                rewards[i] = R_HALLUC
                st["applied_luna_severe"] += 1
        elif restate and statuses[i] == "exact" and rewards[i] > 0:
            rewards[i] = rewards[i] * R_RESTATE_MULT
            st["applied_restate"] += 1
    st["timeouts"] += c["timeout"]
    st["fails"] += c["fail"]
    rec.update(luna=c, luna_reward=luna_reward, judge_wall_s=round(time.time() - t0, 1))
    if quart:
        rec["verdict_by_len_q"] = list(st["verdict_by_len_q"])
    if st["early_calls"]:
        rec["early_calls"] = st["early_calls"]
        rec["judge_span_s"] = round(time.time() - min(h.t_submit for h in handles.values()), 1)
    if st["skipped_rate"] or st["skipped_frac"]:
        rec["skipped"] = {"rate": st["skipped_rate"], "frac": st["skipped_frac"]}
    if c["timeout"]:
        _log({"t": round(time.time(), 1), "halluc_timeout": c["timeout"], "pseudo_step": step_idx})
    return rec


# ------------------------------------------------------------------------- logging
def _log(rec: dict) -> None:
    path = os.environ.get("RLFORGE_TASK_LOG") or os.environ.get("AIQ_TASK_LOG", "")
    if not path:
        return
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    try:
        with _LOG_LOCK, open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass


_SAMPLE_STATE = {"written": 0}


def _log_sample(completions, completion_ids, gold, rewards, source, cap, max_rows=40000):
    path = os.environ.get("AIQ_SAMPLE_LOG") or os.environ.get("RLFORGE_SAMPLE_LOG", "")
    if not path or not rewards:
        return
    try:
        i = random.randrange(len(rewards))
        c = completions[i]
        text = c if isinstance(c, str) else (thinking_text(c) + "\n</think>\n\n" + after_think_text(c))
        line = json.dumps({"t": round(time.time(), 1), "source": str(source), "gold": str(gold).strip(),
                           "task": "ranking" if is_ranking_gold(gold) else "mcq",
                           "reward": rewards[i], "n_tokens": len(completion_ids[i]),
                           "truncated": len(completion_ids[i]) >= cap, "completion": text},
                          ensure_ascii=False) + "\n"
        # one locked write per row: a 16k-token row is several write() syscalls, which two
        # concurrent groups would otherwise interleave into corrupt JSONL
        with _LOG_LOCK:
            if _SAMPLE_STATE["written"] >= max_rows:
                return
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
            _SAMPLE_STATE["written"] += 1
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------- TRL entry
def think_reward(completions, prompts, completion_ids, answer, cap=16384, **kwargs):
    """TRL reward function (module-level, picklable). One call = one group. Thread-safe (v3):
    rlforge's concurrent score loop may run several groups at once.
    Optional rlforge kwargs: rlforge_info=[dict]*n (group_id in; judged out),
    rlforge_early=[handle|marker|None]*n (judge calls the early hook already submitted)."""
    n = len(completions)
    sources = kwargs.get("source") or ["?"] * n
    qids = kwargs.get("question_id") or [None] * n
    nch = kwargs.get("num_choices") or [None] * n
    info = (kwargs.get("rlforge_info") or [None])[0]
    info = info if isinstance(info, dict) else None
    early = kwargs.get("rlforge_early")
    rewards, statuses, invs = [], [], []
    for comp, ids, gold, k in zip(completions, completion_ids, answer, nch):
        s = score_one(comp, gold, n_tokens=len(ids), cap=cap, num_choices=k)
        rewards.append(s["reward"])
        statuses.append(s["status"])
        invs.append(s["inv"])

    halluc = None
    if _halluc_enabled() and n:
        gid = None
        if info is not None and os.environ.get("AIQ_HALLUC_SELECT", "auto") != "scored":
            gid = info.get("group_id")
        selected, step_idx = _halluc_slot(gid)
        local = _local_stats()
        try:
            halluc = _apply_halluc(rewards, statuses, completions, prompts, selected, local, step_idx, early,
                                   lengths=[len(x) for x in completion_ids])
        except Exception as e:  # noqa: BLE001 - the hook must never break scoring
            halluc = {"error": type(e).__name__}
        finally:
            _close_group(step_idx, local)
        if info is not None:
            info["judged"] = bool(selected and local["judge_calls"] > 0)

    ranking = bool(n) and is_ranking_gold(answer[0])
    src = str(sources[0]) if n else "?"
    parsed_inv = [v for v in invs if v is not None]
    rec = {
        "t": round(time.time(), 1),
        "source": src,
        "qid": qids[0] if n else None,
        "task": "ranking" if ranking else "mcq",
        "n": n,
        "exact": statuses.count("exact"),
        "wrong": statuses.count("wrong"),
        "unparsed": statuses.count("unparsed"),
        "trunc": statuses.count("trunc"),
        "inv": invs if ranking else None,
        "mean_inv": round(sum(parsed_inv) / len(parsed_inv), 4) if parsed_inv else None,
        "reward_mean": round(sum(rewards) / n, 4) if n else None,
        "zero_var": bool(n) and (max(rewards) - min(rewards) < 1e-9),
        "mean_tokens": round(sum(len(x) for x in completion_ids) / n, 1) if n else None,
        # legacy fields (rlforge parsers / dashboard)
        "n_mcq": 0 if ranking else n, "n_rank": n if ranking else 0,
        "mcq_correct": 0 if ranking else statuses.count("exact"),
        "mcq_unparsed": 0 if ranking else statuses.count("unparsed"),
        "mcq_trunc": 0 if ranking else statuses.count("trunc"),
        "rank_exact": statuses.count("exact") if ranking else 0,
        "rank_unparsed": statuses.count("unparsed") if ranking else 0,
        "rank_trunc": statuses.count("trunc") if ranking else 0,
        "rank_score_sum": round(sum(rewards), 4) if ranking else 0.0,
        "by_source": {src: [n, round(sum(rewards), 4), statuses.count("exact"), statuses.count("trunc")]},
    }
    if info is not None:
        rec["group_id"] = info.get("group_id")
    if halluc is not None:
        rec["halluc"] = halluc
    _log(rec)
    _log_sample(completions, completion_ids, answer[0] if n else "", rewards, src, cap)
    return rewards


def rlforge_on_rollout(group_id, prompt, completion, completion_ids, cap=16384, **_):
    """rlforge early hook: runs on the rollout event loop right after ONE rollout finished
    generating; must not block. If the rollout's group is selected for the judge (dispatch-order
    selection, same picks think_reward will use) and it is not truncated, submit its judge call
    now so the group's own generation tail hides the judge latency. AIQ_HALLUC_EARLY=0 disables."""
    if not _halluc_enabled() or os.environ.get("AIQ_HALLUC_EARLY", "1") != "1":
        return None
    if os.environ.get("AIQ_HALLUC_SELECT", "auto") == "scored" or not _is_selected_gid(group_id):
        return None
    if len(completion_ids) >= cap:  # truncated: already -2, never judged
        return None
    if not _frac_ok():
        return _SKIP_FRAC
    if not _admit():
        return _SKIP_RATE
    return _submit_judge(_prompt_text(prompt), thinking_text(completion), final_answer_body(completion) or "")


think_reward.rlforge_on_rollout = rlforge_on_rollout
