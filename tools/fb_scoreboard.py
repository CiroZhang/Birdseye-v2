"""Read the broadcast scoreboard of FBBench / FineBadminton clips (RapidOCR, CPU): the two players' current
game points over time. Used for rally counting (each rally ends in exactly one point).

Scoreboard rows: a name and one or more numbers (earlier games first, current game last). Per sampled frame,
OCR the top-left corner, group the text into rows, take each row's rightmost number -> (a, b); None when the
board is hidden (replays, transitions) or unreadable.

Output: outputs/<set>/scoreboard/<uid>.json  {fps, step, reads: [[frame, a, b] or [frame, null, null]]}
usage: python tools/fb_scoreboard.py [uid ...]   (env FB_SET=bench, STEP_S=2, WORKERS=8)"""
import glob, json, os, re, sys

import numpy as np
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FB_SET = os.environ.get("FB_SET", "bench")
CLIPS = os.path.join(ROOT, "data", *{"bench": ("fbbench", "hf_video_clips_qa"), "train": ("fb20k", "clips")}[FB_SET])
OUT = os.path.join(ROOT, "outputs", {"bench": "fb", "train": "fb20k"}[FB_SET], "scoreboard")
STEP_S = float(os.environ.get("STEP_S", "2"))
CROP = (0.0, 0.0, 0.45, 0.28)  # x0, y0, x1, y1 as a share of the frame: broadcast boards sit top-left

OCR_THREADS = int(os.environ.get("OCR_THREADS", "2"))  # per worker; the default (all cores) starves other jobs
_eng = None


def read_board(img):
    """-> (a, b) current points of the upper and lower row, or None."""
    global _eng
    if _eng is None:
        import logging
        logging.disable(logging.WARNING)
        from rapidocr import RapidOCR
        _eng = RapidOCR(params={"EngineConfig.onnxruntime.intra_op_num_threads": OCR_THREADS,
                                "EngineConfig.onnxruntime.inter_op_num_threads": 1})
    h, w = img.shape[:2]
    crop = img[int(CROP[1] * h):int(CROP[3] * h), int(CROP[0] * w):int(CROP[2] * w)]
    r = _eng(crop)
    if r.txts is None:
        return None
    toks = []
    for box, t, sc in zip(r.boxes, r.txts, r.scores):
        if sc < 0.6:
            continue
        b = list(map(list, box))
        toks.append(((min(p[1] for p in b) + max(p[1] for p in b)) / 2, min(p[0] for p in b),
                     max(p[1] for p in b) - min(p[1] for p in b), t.strip()))
    names = [t for t in toks if re.search(r"[A-Za-z]{2,}", t[3])]
    nums = [t for t in toks if re.fullmatch(r"\d{1,2}", t[3])]
    if len(names) < 2 or len(nums) < 2:
        return None
    # the two player rows: the two name tokens left-most / closest in height to each other
    names.sort(key=lambda t: t[0])
    best = None
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        gap = b[0] - a[0]
        if 0.6 * a[2] < gap < 3.0 * a[2] and abs(a[1] - b[1]) < 3 * a[2]:
            best = (a, b)
            break
    if best is None:
        return None
    rows = []
    for nm in best:
        row = [t for t in nums if abs(t[0] - nm[0]) < 0.6 * nm[2] and t[1] > nm[1]]
        if not row:
            return None
        rows.append(int(max(row, key=lambda t: t[1])[3]))
    return tuple(rows) if all(0 <= v <= 30 for v in rows) else None


def run(uid):
    import cv2
    cv2.setNumThreads(1)
    dst = os.path.join(OUT, uid + ".json")
    if os.path.exists(dst):
        return None
    try:
        cap = cv2.VideoCapture(os.path.join(CLIPS, uid + ".mp4"))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(STEP_S * fps)))
        reads = []
        for f in list(range(0, n, step)) + [n - 1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                continue
            ab = read_board(im)
            reads.append([f, *(ab if ab else (None, None))])
        os.makedirs(OUT, exist_ok=True)
        json.dump({"fps": fps, "step": step, "reads": reads}, open(dst, "w"))
        seen = [r for r in reads if r[1] is not None]
        return f"{uid} {len(reads)} reads, {len(seen)} with a board: {seen[0][1:] if seen else None} -> {seen[-1][1:] if seen else None}"
    except Exception as e:
        return f"FAILED {uid} {e!r}"


def game_over(a, b):
    return (max(a, b) >= 21 and abs(a - b) >= 2) or max(a, b) == 30


MIN_POINT_S = 8.0  # s: a rally plus the break before the next serve is longer than this


def _step(u, v, dt_s=None):
    """Points between two board states dt_s seconds apart, or None if v cannot follow u: a score went down
    outside a game reset, or more points than the time allows."""
    if v[0] >= u[0] and v[1] >= u[1]:
        pts = v[0] - u[0] + v[1] - u[1]
    elif game_over(*u) and v[0] + v[1] <= 10:
        pts = 1 + v[0] + v[1]
    else:
        return None
    if dt_s is not None and pts > 1 + dt_s / MIN_POINT_S:
        return None
    return pts


def rallies(reads, fps=30.0):
    """Points scored while the board is visible, +1 for the rally being played at the last reading (the board
    shows the score before it). Misreads (another number on the row, an earlier game's 21, a digit dropped) are
    outliers: keep the longest chain of readings, in time order, along which the score never goes down (game
    resets allowed after a game-ending score), and count the points along it."""
    obs = [((a, b), f) for f, a, b in reads if a is not None]
    if not obs:
        return None, {}
    n = len(obs)
    best, prev = [1] * n, [-1] * n
    for k in range(n):
        for i in range(k):
            if best[i] + 1 > best[k] and _step(obs[i][0], obs[k][0], (obs[k][1] - obs[i][1]) / fps) is not None:
                best[k], prev[k] = best[i] + 1, i
    k = int(np.argmax(best))
    chain = []
    while k >= 0:
        chain.append(obs[k]); k = prev[k]
    chain = chain[::-1]
    pts = sum(_step(u[0], v[0]) for u, v in zip(chain[:-1], chain[1:]))
    return pts + 1, {"first": list(chain[0][0]), "last": list(chain[-1][0]), "first_frame": chain[0][1],
                     "last_frame": chain[-1][1], "kept": len(chain), "read": n}


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    uids = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CLIPS, "*.mp4")))
    with Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for msg in pool.imap_unordered(run, uids):
            if msg:
                print(msg, flush=True)
