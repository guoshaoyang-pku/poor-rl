#!/usr/bin/env python3
"""Raw TCP throughput between two hosts (no ssh between hosts needed).

server: python3 tcp_bench.py server --bind 172.16.0.15 --port 29660 --streams 8 --seconds 20
client: python3 tcp_bench.py client --host 172.16.0.15 --port 29660 --streams 8 --seconds 15
Each stream uses port+i. Client prints one JSON line with Gbit/s (sent bytes / wall time).
Server exits after all streams close (or --timeout).
"""
import argparse, json, os, socket, threading, time

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["server", "client"])
p.add_argument("--bind", default="0.0.0.0")
p.add_argument("--host", default="")
p.add_argument("--port", type=int, default=29660)
p.add_argument("--streams", type=int, default=1)
p.add_argument("--seconds", type=float, default=10)
p.add_argument("--buf", type=int, default=4 << 20, help="application send/recv chunk size")
# [review fix] setting SO_RCVBUF/SO_SNDBUF explicitly DISABLES Linux TCP autotuning, and in these containers the 4 MiB
# request is capped to 416 KiB (2 x rmem_max) -> per-stream window 416 KiB instead of autotuned 6 MiB (tcp_rmem max),
# which under-reports what NCCL NET/Socket (which keeps kernel defaults) can reach. Default now: leave kernel defaults.
p.add_argument("--sockbuf", type=int, default=0, help="if >0, setsockopt SO_RCVBUF/SO_SNDBUF to this (disables autotune)")
p.add_argument("--timeout", type=float, default=120)
a = p.parse_args()

if a.mode == "server":
    tot = [0] * a.streams
    def serve(i):
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((a.bind, a.port + i)); s.listen(1); s.settimeout(a.timeout)
        c, _ = s.accept()
        if a.sockbuf > 0:
            c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, a.sockbuf)
        b = bytearray(a.buf); mv = memoryview(b)
        while True:
            n = c.recv_into(mv)
            if n == 0: break
            tot[i] += n
        c.close(); s.close()
    ts = [threading.Thread(target=serve, args=(i,), daemon=True) for i in range(a.streams)]
    [t.start() for t in ts]; [t.join(a.timeout) for t in ts]
    print(json.dumps({"role": "server", "bytes": sum(tot)}), flush=True)
else:
    payload = os.urandom(a.buf)
    sent = [0] * a.streams
    t_end = [0.0]
    def go(i):
        c = socket.create_connection((a.host, a.port + i), timeout=30)
        if a.sockbuf > 0:
            c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, a.sockbuf)
        dl = time.time() + a.seconds
        while time.time() < dl:
            c.sendall(payload); sent[i] += len(payload)
        c.shutdown(socket.SHUT_WR); c.close()
    t0 = time.time()
    ts = [threading.Thread(target=go, args=(i,)) for i in range(a.streams)]
    [t.start() for t in ts]; [t.join() for t in ts]
    dt = time.time() - t0
    print(json.dumps({"role": "client", "streams": a.streams, "sockbuf": a.sockbuf, "seconds": round(dt, 2), "bytes": sum(sent),
                      "gbit_s": round(sum(sent) * 8 / dt / 1e9, 2)}), flush=True)
