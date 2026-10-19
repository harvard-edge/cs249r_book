"""Real-socket blind-retry experiment on localhost.
Server: one HTTP endpoint POST /restart. Each delivered request is an attempt.
With probability p_req the server drops the request (closes without executing,
no reply) = request loss. Otherwise it executes (counts one effect for the
operation); then with probability p_rep it withholds the reply past the client
timeout = reply loss. With dedup=True the server keeps op_id -> result and a
repeated op_id executes nothing (Idempotency-Key style).
Client: each operation gets an op id; on timeout it retries exactly once.
Effects per op counted at the server. Losses drawn by a seeded RNG per attempt.
"""
import http.server, http.client, urllib.error, threading, random, sys, time, urllib.request, json, socket
from collections import Counter
def run(n, p_req, p_rep, dedup, seed=7, timeout=0.05):
    rng = random.Random(seed); lock = threading.Lock()
    effects = Counter(); seen = {}
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            op = self.headers["Idempotency-Key"]
            with lock:
                drop_req = rng.random() < p_req; drop_rep = rng.random() < p_rep
            if drop_req:
                time.sleep(timeout * 2); self.close_connection = True; return
            with lock:
                if dedup and op in seen: pass
                else: effects[op] += 1; seen[op] = "ok"
            if drop_rep: time.sleep(timeout * 2); self.close_connection = True; return
            self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/restart"
    timeouts = 0
    def call(op):
        req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Idempotency-Key": op})
        try:
            urllib.request.urlopen(req, timeout=timeout).read(); return True
        except (socket.timeout, TimeoutError, urllib.error.URLError, ConnectionError, http.client.HTTPException):
            return False

    for i in range(n):
        op = f"op-{i}"
        if not call(op):
            timeouts += 1; call(op)
    srv.shutdown()
    dups = sum(1 for v in effects.values() if v > 1)
    return dict(n=n, p_req=p_req, p_rep=p_rep, dedup=dedup, timeouts=timeouts, duplicates=dups,
                executed_ops=len(effects))
if __name__ == "__main__":
    res = []
    for (n, pq, pr, dd) in [(20000, .02, .02, False), (20000, .02, .01, False), (20000, .02, .04, False), (20000, .02, .02, True)]:
        r = run(n, pq, pr, dd); print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
