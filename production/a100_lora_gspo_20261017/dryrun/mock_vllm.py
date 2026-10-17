#!/usr/bin/env python3
"""Dry-drill mock of the vLLM endpoints that the router, head.py and resume_watch.py use, on N ports from BASE.
Each port starts with trl-policy-v57..v60 loaded, as left by a dead earlier router. Loading a name that is already
loaded answers 400 (like vLLM); unloading an unknown one answers 404. Every POST is logged to LOG.
Usage: mock_vllm.py LOG BASE N
"""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG, BASE, N = open(sys.argv[1], 'a', buffering=1), int(sys.argv[2]), int(sys.argv[3])
lock = threading.Lock()


def server(port):
    loaded = {f'trl-policy-v{v}' for v in (57, 58, 59, 60)}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/health':
                return self.reply(200, {})
            if path == '/v1/models':
                with lock:
                    names = sorted(loaded)
                return self.reply(200, {'object': 'list', 'data': [{'id': 'base', 'parent': None}] +
                                        [{'id': n, 'parent': 'base'} for n in names]})
            if path == '/server_info':
                return self.reply(200, {'vllm_config': {'lora_config': {'max_loras': 5, 'max_lora_rank': 32},
                                                        'parallel_config': {'data_parallel_size': 1}}})
            self.reply(404, {})

        def do_POST(self):
            path = self.path.split('?')[0]
            name = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}').get('lora_name')
            with lock:
                code = 200
                if path == '/v1/load_lora_adapter':
                    code = 400 if name in loaded else 200; loaded.add(name)
                elif path == '/v1/unload_lora_adapter':
                    code = 200 if name in loaded else 404; loaded.discard(name)
                LOG.write(json.dumps({'port': port, 'path': path, 'name': name, 'code': code}) + '\n')
            self.reply(code, {})

    return ThreadingHTTPServer(('127.0.0.1', port), H)


servers = [server(BASE + i) for i in range(N)]
for s in servers[1:]:
    threading.Thread(target=s.serve_forever, daemon=True).start()
servers[0].serve_forever()
