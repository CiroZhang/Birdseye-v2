"""Download Yangshipin / CCTV mp4 links (obtained by hand, one per view) robustly: the file is fetched in 32 MB pieces over
several connections; each finished piece is kept on disk, so a dropped connection or a restart only re-fetches the
piece it was on. Pieces are joined and the size checked at the end.

usage: python tools/cctv_fetch.py <out_dir> [<name>=]<url> [...]   (env CONN = connections per file, default 8)
  -> <out_dir>/<name or video id>.mp4  (progress: one line per file every 30 s)"""
import os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import requests

PIECE = 32 << 20
CONN = int(os.environ.get("CONN", "8"))
UA = {"User-Agent": "Mozilla/5.0"}


def size_of(url):
    r = requests.get(url, headers={**UA, "Range": "bytes=0-0"}, timeout=30)
    r.raise_for_status()
    return int(r.headers["Content-Range"].split("/")[1])


def fetch_piece(url, path, a, b, done):
    want = b - a + 1
    for attempt in range(1000):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == want:
            return
        if have > want:
            os.remove(path)
            have = 0
        try:
            with requests.get(url, headers={**UA, "Range": f"bytes={a + have}-{b}"}, stream=True, timeout=(20, 60)) as r:
                if r.status_code in (403, 410):
                    raise SystemExit(f"link refused ({r.status_code}): expired? get a fresh link")
                r.raise_for_status()
                with open(path, "ab") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                        done[0] += len(chunk)
        except SystemExit:
            raise
        except Exception as e:  # dropped / timed out: resume this piece from what it has
            time.sleep(min(5 * (attempt + 1), 60))


def fetch(arg, out_dir):
    name, url = arg.split("=", 1) if not arg.startswith("http") else ("", arg)
    name = (name or url.split("?")[0].rsplit("/", 1)[1].split(".")[0]) + ".mp4"
    dst = os.path.join(out_dir, name)
    total = size_of(url)
    if os.path.exists(dst) and os.path.getsize(dst) == total:
        print(name, "already complete", flush=True)
        return
    parts = dst + ".parts"
    os.makedirs(parts, exist_ok=True)
    spans = [(a, min(a + PIECE, total) - 1) for a in range(0, total, PIECE)]
    paths = [os.path.join(parts, f"{i:05d}") for i in range(len(spans))]
    done = [sum(os.path.getsize(p) for p in paths if os.path.exists(p))]
    stop = threading.Event()

    def report():
        last, t0 = done[0], time.time()
        while not stop.wait(30):
            now = done[0]
            print(f"{name} {now / total * 100:5.1f}%  {now / 2**30:.2f}/{total / 2**30:.2f} GB  "
                  f"{(now - last) / 30 / 2**20:.1f} MB/s", flush=True)
            last = now

    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(CONN) as ex:
        list(ex.map(lambda i: fetch_piece(url, paths[i], *spans[i], done), range(len(spans))))
    stop.set()
    with open(dst + ".tmp", "wb") as f:
        for p in paths:
            with open(p, "rb") as g:
                while True:
                    buf = g.read(1 << 24)
                    if not buf:
                        break
                    f.write(buf)
    if os.path.getsize(dst + ".tmp") != total:
        raise SystemExit(f"{name}: size mismatch, keeping pieces")
    os.replace(dst + ".tmp", dst)
    for p in paths:
        os.remove(p)
    os.rmdir(parts)
    print(name, "DONE", f"{total / 2**30:.2f} GB", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    with ThreadPoolExecutor(len(sys.argv) - 2) as ex:
        list(ex.map(lambda u: fetch(u, out), sys.argv[2:]))
    print("ALL DONE", flush=True)
