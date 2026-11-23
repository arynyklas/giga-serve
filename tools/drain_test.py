"""SIGTERM drain check: N requests in flight, then the server gets SIGTERM.

  drain_test.py URL MODEL N WORDS   (run next to the server; it signals the giga_serve.py process)
Every accepted request must come back 200 with all its vectors; a request sent after SIGTERM
must be refused at connect; the server process must exit by itself.
"""
import concurrent.futures as cf
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

URL, MODEL, N, WORDS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rnd = random.Random(11)
WORDLIST = "погода город river book music 2025 coffee".split()
texts = [" ".join(rnd.choice(WORDLIST) for _ in range(WORDS)) for _ in range(64)]


def post(i):
    t0 = time.time()
    body = json.dumps({"model": MODEL, "input": texts, "encoding_format": "base64"}).encode()
    req = urllib.request.Request(URL + "/v1/embeddings", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            n = len(json.loads(r.read())["data"])
            return i, r.status, n, round(time.time() - t0, 2)
    except urllib.error.HTTPError as e:
        return i, e.code, 0, round(time.time() - t0, 2)
    except Exception as e:  # noqa: BLE001
        return i, repr(e)[:80], 0, round(time.time() - t0, 2)


pid = int(subprocess.check_output(["pgrep", "-f", "^python3 giga_serve.py"]).split()[0])
with cf.ThreadPoolExecutor(N + 1) as ex:
    futs = [ex.submit(post, i) for i in range(N)]
    time.sleep(1.0)
    t_sig = time.time()
    os.kill(pid, signal.SIGTERM)
    time.sleep(0.5)
    late = ex.submit(post, -1)
    res = [f.result() for f in futs]
    late_res = late.result()
t_exit = None
for _ in range(600):
    if subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode != 0:
        t_exit = time.time()
        break
    time.sleep(0.5)
ok = sum(1 for r in res if r[1] == 200 and r[2] == 64)
print(json.dumps({"in_flight": N, "ok_200_full": ok, "statuses": sorted({str(r[1]) for r in res}),
                  "last_answer_s_after_sigterm": round(max(r[3] for r in res) - 1.0, 2),
                  "late_request": late_res[1], "exit_s_after_sigterm": round(t_exit - t_sig, 2) if t_exit else None}))
