"""Recon for the token-policy prototype (SFT stage).

Q1: how do the 128 canonical decimal answers ("0.004".."0.996", format spec v1)
    tokenize? If the fractional part is one token, the policy is an EXACT
    softmax over <=128 single tokens at one position -> cleanest possible
    "token probability == normalized Q". Otherwise we supervise digit by digit.
Q2: digit/dot token ids, chat-template tail, tie_word_embeddings, lm_head layout.
Q3: bc_collector shard keys/shapes (for the SFT dataset).
"""
import json
import os
import sys
from collections import Counter

import numpy as np


def main():
    mp = sys.argv[1]
    data = sys.argv[2] if len(sys.argv) > 2 else None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mp)

    strs = [f"{(c + 0.5) / 128:.3f}" for c in range(128)]
    enc = {s: tok(s, add_special_tokens=False)["input_ids"] for s in strs}
    print("[Q1] token-length distribution of the 128 answer strings:",
          dict(Counter(len(v) for v in enc.values())))
    toks_per_str = Counter(tuple(v) for v in enc.values())
    print("[Q1] distinct encodings:", len(toks_per_str))
    print("[Q1] most common encodings:")
    for k, n in toks_per_str.most_common(3):
        print("     ", list(k), "->", repr(tok.decode(list(k))), "x", n)
    for s in ("0.004", "0.012", "0.293", "0.996"):
        if s in enc:
            print(f"[Q1] {s} -> {enc[s]} = {[tok.decode([i]) for i in enc[s]]!r}")
    print("[Q1] first-token set:", len(set(v[0] for v in enc.values())),
          set(v[0] for v in enc.values()))
    print("[Q1] prefix2 set:", len(set(tuple(v[:2]) for v in enc.values())),
          set(tuple(v[:2]) for v in enc.values()))

    print("[Q2] digit ids:", {str(d): tok.encode(str(d), add_special_tokens=False)
                              for d in range(10)})
    print("[Q2] dot id:", tok.encode(".", add_special_tokens=False))
    print("[Q2] '0.' id:", tok.encode("0.", add_special_tokens=False))
    print("[Q2] '0.004' full:", tok.encode("0.004", add_special_tokens=False))
    print("[Q2] eos/pad:", tok.eos_token, tok.eos_token_id, tok.pad_token,
          tok.pad_token_id)
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("[Q2] template tail:", repr(t[-90:]))
    p = os.path.join(mp, "config.json")
    cfg = json.load(open(p)) if os.path.exists(p) else {}
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    print("[Q2] tie_word_embeddings:", cfg.get("tie_word_embeddings"),
          "hidden:", cfg.get("hidden_size"), "vocab:", cfg.get("vocab_size"))

    if data:
        fs = sorted(f for f in os.listdir(data) if f.endswith(".npz"))[:1]
        if fs:
            z = np.load(os.path.join(data, fs[0]))
            print("[Q3] shard keys:", {k: (z[k].shape, str(z[k].dtype))
                                       for k in z.files})


if __name__ == "__main__":
    main()
