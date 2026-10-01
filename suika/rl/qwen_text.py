"""Text serialization for the Qwen LLM arm — format spec v1 (FROZEN).

Compact obs layout = encoding.encode_tokens: [T,5] rows (type, x_n, y_n, vx_n,
vy_n); row 0 = current fruit, row 1 = next fruit, rows 2.. = board fruits
sorted by (y, x) ascending (top / killy-danger first); type < 0 = padding.

Prompt (chat template applied at tokenization):
  system: 你在玩合成大西瓜。给定盘面，输出当前水果的横向落点，一个 0 到 1 的小数，只输出数字。
  user:   当前果:3 下一果:1\n盘面: 3 42 18 | 3 45 21 | ...

- board coords are ints in [0,127]; y=0 at the top (killy side)
- at most n_text board fruits serialized; truncation drops the LOWEST rows
  (bottom of the stack = safest to omit), top rows always kept
- velocities are NOT serialized (settle mode: near-still at decision time;
  revisit for tempo)
- mirror aug happens on the compact obs (learner._mirror tokens branch),
  then re-serialization — x ints flip as 127-x, consistent by construction

Decimal action interface (display layer over the 128-bin Q head):
  col -> "0.XXX" with x = (col + 0.5) / K;   col = floor(x * K), clamped.
The model itself never generates text in nothink mode; these helpers exist
for logs / demos / the later think-mode phase.
"""
import numpy as np

SYSTEM_PROMPT = ("你在玩合成大西瓜。给定盘面，输出当前水果的横向落点，"
                 "一个 0 到 1 的小数，只输出数字。")


def build_user_text(compact, n_text=96):
    """compact: [T,5] or [T*5] float array -> user-turn string."""
    tok = np.asarray(compact, dtype=np.float32).reshape(-1, 5)
    cur, nxt = int(tok[0, 0]), int(tok[1, 0])
    rows = []
    for r in tok[2:]:
        t = r[0]
        if t < 0:
            break
        rows.append((int(t), int(round(float(r[1]) * 127.0)),
                     int(round(float(r[2]) * 127.0))))
        if len(rows) >= n_text:
            break
    board = " | ".join(f"{t} {x} {y}" for t, x, y in rows) if rows else "空"
    return f"当前果:{cur} 下一果:{nxt}\n盘面: {board}"


def build_messages(compact, n_text=96):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_text(compact, n_text)},
    ]


def encode_prompt_batch(compacts, tokenizer, n_text=96, max_len=640):
    """compacts: [B, T*5] float32 ndarray -> padded ids/attention_mask tensors.

    Deterministic; the single tokenization path shared by learner / inference
    server / evaluator (they all call it through QwenQ.forward).
    """
    texts = [tokenizer.apply_chat_template(
        build_messages(c, n_text), tokenize=False, add_generation_prompt=True)
        for c in compacts]
    return tokenizer(texts, padding=True, truncation=True, max_length=max_len,
                     return_tensors="pt", add_special_tokens=False)


def col_to_decimal(col, K=128):
    """128-bin action -> canonical display decimal '0.XXX'."""
    return f"{(int(col) + 0.5) / int(K):.3f}"


def decimal_to_col(text, K=128):
    """Parse '0.XXX' (or any float string) -> bin index."""
    x = float(str(text).strip())
    return int(min(int(K) - 1, max(0, np.floor(x * int(K)))))
