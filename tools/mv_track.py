"""Multi-camera player tracks: the 2 (singles) or 4 (doubles) players on the court floor, from every camera.

Per TV frame, the floor points of every camera's pose detections (tools/mv_people.py) are merged into people: the
player cameras first (they are framed on the players), then the side camera, then the TV; each camera adds at most
one detection to a person (Hungarian on floor distance, gate GATE). A person's floor point is the precision-weighted
mean of its detections. Players are then followed through the rally: each keeps its side of the net, each frame is
assigned by Hungarian on distance to the player's last position (gate grows while unseen), people seen by a player
camera cost less (they are players, not officials or ball crew). Singles or doubles from how many people the player
cameras see. Started from the frame where the player cameras see the most players, then run forward and backward.

Output outputs/cctv/mv_track/<uid>.npz: foot (T,K,2) metres (NaN unseen), views (T,K) bitmask tv=1 side=2 p1=4 p2=8,
<view>_idx (T,K) index of that camera's pose detection in the person (-1 none; camera frame per mv_people), K.
usage: python tools/mv_track.py uid ...   (all rallies with a TV calibration when no uid is given)"""
import glob, os, sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import mv_people as MP  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "cctv", "mv_track")
NET_Y, CX = 6.7, 3.05
GATE = 0.8                                   # m: two cameras' floor points of one person
SIG = {"p1": 0.15, "p2": 0.15, "side": 0.2, "tv": 0.35}   # m: floor-point noise per camera (weights of the mean)
BIT = {"tv": 1, "side": 2, "p1": 4, "p2": 8}
ORDER = ("p1", "p2", "side", "tv")
NEAR_COURT = lambda xy: abs(xy[0] - CX) < 4.5 and -2.5 < xy[1] < 15.9


def merge(people):
    """{view: [(xy, kp, h)]} -> list of persons {xy, views bitmask, idx {view: detection index}}"""
    P = []
    for v in ORDER:
        pts = [(i, p[0]) for i, p in enumerate(people.get(v, [])) if NEAR_COURT(p[0])]
        if not pts:
            continue
        if P:
            C = np.array([[np.linalg.norm(q["xy"] - xy) for q in P] for _, xy in pts])
            r, c = linear_sum_assignment(C)
            done = set()
            for a, b in zip(r, c):
                if C[a, b] < GATE:
                    q = P[b]; w = 1 / SIG[v] ** 2
                    q["xy"] = (q["xy"] * q["w"] + pts[a][1] * w) / (q["w"] + w); q["w"] += w; q["views"] |= BIT[v]
                    q["idx"][v] = pts[a][0]
                    done.add(a)
            new = [pts[a] for a in range(len(pts)) if a not in done]
        else:
            new = pts
        for i, xy in new:
            P.append({"xy": np.array(xy, float), "w": 1 / SIG[v] ** 2, "views": BIT[v], "idx": {v: i}})
    return P


def n_players(frames):
    """4 when the cameras usually agree on more than two people on the court (doubles), else 2. Player cameras may
    follow one player each even in doubles, so the count is over people seen by at least two cameras."""
    inside = lambda xy: abs(xy[0] - CX) < 3.6 and -1.0 < xy[1] < 14.4
    c = [sum(1 for q in P if bin(q["views"]).count("1") >= 2 and inside(q["xy"])) for P in frames]
    c = [x for x in c if x > 0]
    return 4 if c and np.percentile(c, 75) >= 3 else 2


def cost(track_xy, unseen, q):
    d = np.linalg.norm(track_xy - q["xy"])
    gate = min(1.0 + 0.25 * unseen, 4.0)
    if d > gate:
        return 1e6
    return d - 0.4 * bool(q["views"] & 12) - 0.1 * bin(q["views"]).count("1")


def run_dir(frames, order, init, K):
    """follow K players over frames in the given order from initial positions; -> {f: [person index or -1]*K}"""
    pos = [np.array(x, float) for x in init]; side = [x[1] > NET_Y for x in init]; unseen = [0] * K
    out = {}
    for f in order:
        P = frames[f]
        if not P:
            out[f] = [-1] * K; unseen = [u + 1 for u in unseen]; continue
        C = np.full((K, len(P)), 1e6)
        for k in range(K):
            for j, q in enumerate(P):
                if (q["xy"][1] > NET_Y) == side[k] or abs(q["xy"][1] - NET_Y) < 0.3:
                    C[k, j] = cost(pos[k], unseen[k], q)
        r, c = linear_sum_assignment(C)
        a = [-1] * K
        for k, j in zip(r, c):
            if C[k, j] < 1e5:
                a[k] = j
        for k in range(K):
            if a[k] >= 0:
                pos[k] = P[a[k]]["xy"]; unseen[k] = 0
            else:
                unseen[k] += 1
        out[f] = a
    return out


def track(uid):
    pe = MP.People(uid)
    T = len(pe.views["tv"][2])
    frames = [merge(pe.people(f)) for f in range(T)]
    K = n_players(frames)
    # start: frame where the player cameras see K people, K/2 per side, best supported
    best, f0 = (-1, -1), None
    for f, P in enumerate(frames):
        pl = [q for q in P if q["views"] & 12]
        nf = sum(q["xy"][1] > NET_Y for q in pl)
        sc = (int(len(pl) == K and nf == K // 2), sum(bin(q["views"]).count("1") for q in pl))   # balanced first
        if sc > best:
            best, f0 = sc, f
    if f0 is None:                                   # no player cameras: the best-supported frame
        f0 = max(range(T), key=lambda f: len(frames[f]))
    pl = sorted([q for q in frames[f0] if q["views"] & 12] or frames[f0], key=lambda q: -bin(q["views"]).count("1"))
    near = [q for q in pl if q["xy"][1] <= NET_Y][:K // 2]; far = [q for q in pl if q["xy"][1] > NET_Y][:K // 2]
    init = [q["xy"] for q in near] + [q["xy"] for q in far]
    while len(init) < K:                             # a side short of players at the start: placeholder on that side
        init.append(np.array([CX, 3.0 if sum(x[1] <= NET_Y for x in init) < K // 2 else 10.4]))
    fw = run_dir(frames, range(f0, T), init, K)
    bw = run_dir(frames, range(f0, -1, -1), init, K)
    foot = np.full((T, K, 2), np.nan, np.float32); views = np.zeros((T, K), np.int8)
    idx = {v: np.full((T, K), -1, np.int16) for v in ORDER}
    for f in range(T):
        a = fw[f] if f >= f0 else bw[f]
        for k, j in enumerate(a):
            if j >= 0:
                q = frames[f][j]; foot[f, k] = q["xy"]; views[f, k] = q["views"]
                for v, i in q["idx"].items():
                    idx[v][f, k] = i
    return dict(foot=foot, views=views, K=K, **{v + "_idx": a for v, a in idx.items()})


def main():
    uids = sys.argv[1:] or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ROOT, "outputs", "cctv", "calib_bc", "*.json")))
    os.makedirs(OUT, exist_ok=True)
    for u in uids:
        try:
            o = track(u)
            np.savez_compressed(os.path.join(OUT, u + ".npz"), **o)
            seen = np.isfinite(o["foot"][..., 0]).mean()
            print(u, "K", o["K"], f"seen {seen:.1%}", flush=True)
        except Exception as e:
            print(u, "FAILED", repr(e), flush=True)


if __name__ == "__main__":
    main()
