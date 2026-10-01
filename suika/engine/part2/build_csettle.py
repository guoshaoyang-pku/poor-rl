#!/usr/bin/env python
"""Compile part2/csettle.c into _csettle_c.so for the current platform.

Run once per machine/venv that runs actors or evaluation:
    python build_csettle.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "csettle.c")
OUT = os.path.join(HERE, "_csettle_c.so")

candidates = [c for c in (os.environ.get("CC"), "cc", "gcc", "clang") if c]
for cc in candidates:
    try:
        r = subprocess.run([cc, "-O2", "-fPIC", "-shared", "-std=c99",
                            SRC, "-o", OUT])
    except FileNotFoundError:
        continue
    if r.returncode == 0:
        print("built", OUT, "with", cc)
        sys.exit(0)

sys.exit("no working C compiler (tried: %s)" % ", ".join(candidates))
