"""Run the two-view physics teacher (birdseye_v2/teacher_dense.py) over the CCTV training rallies.

Each rally: the TV and fixed side camera's refined shuttle tracks on the TV frame clock (side frame = TV frame + the
session's shift, tools/cctv_tri.py), the triangulated points as initial guesses.
Output: outputs/cctv/teach/<uid>.json {frames, P, V, sd, hits, ..., S: [[frame, x, y, z], ...] for the confident frames
only (the student's labels)}. Held-out test matches (outputs/cctv/split.json) are skipped.
usage: python tools/cctv_teach.py [uid ...]   (env NPROC, default 12)"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "outputs", "cctv", os.environ.get("T_OUT", "teach"))
FPS = 25.0


def load(uid):
    """-> (Views on the TV frame clock, triangulated points (n,4), player context for the teacher's priors)."""
    from birdseye_v2.court import Camera
    from birdseye_v2 import teacher as TE, shuttle3d as S3
    import cctv_tri as CT
    S3.FPS = FPS
    tri = json.load(open(os.path.join(ROOT, "outputs", "cctv", "tri", uid + ".json")))
    side, _ = CT.side_camera(uid)
    tv = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "cctv", "calib_bc", uid + ".json")))["cam"])
    a = CT.track(os.path.join(ROOT, "outputs", "cctv", "refined", uid + ".json"))
    b = CT.track(os.path.join(ROOT, "outputs", "cctv_side", "refined", uid + ".json"))
    if len(a) + len(b) < 20 or not b:
        raise ValueError("too few detections" if b else "no side-camera detections")
    fa, fb = np.array(sorted(a), float), np.array(sorted(b), float)
    ctx = {}
    ob = os.path.join(ROOT, "outputs", "rally_obs", "cctv", uid + ".npz")
    if os.path.exists(ob):
        o = np.load(ob)
        ctx = {"foot": o["foot"], "C": o["C"], "rk_dir": o["rk_dir"], "rk_conf": o["rk_conf"]}
    V = TE.Views([tv, side], [fa, fb - tri["shift"]], [np.array([a[k] for k in sorted(a)]), np.array([b[k] for k in sorted(b)])], **ctx)
    return V, np.array(tri["shuttle"], float).reshape(-1, 4), ctx


GAP = float(os.environ.get("T_GAP", "0.2"))  # m: a player-camera detection is taken when its ray passes this close to a TV or side ray
SELF_PX = float(os.environ.get("T_SELF_PX", "20"))  # px: second pass, a detection this close to the teacher's own projected path


def _gap(c1, d1, c2, d2):
    """Closest approach of two rays -> (gap m, midpoint, depth1, depth2)."""
    w = c1 - c2
    a, b, c, d, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-9:
        return np.inf, None, -1, -1
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    p1, p2 = c1 + s * d1, c2 + t * d2
    return float(np.linalg.norm(p1 - p2)), (p1 + p2) / 2, s, t


PCAM_SYNC = {"p1": -7.5, "p2": -6.9}  # pcam frame k shows TV time k - sync (fractional scan, 30 dev rallies, sf)


def pcam_candidates(uid):
    """Per player camera with a calibration and stabilised detections: [(TV frame t, peaks (m,2), R, f)], C, k1."""
    from scipy.spatial.transform import Rotation
    out = {}
    for v in ("p1", "p2"):
        cp = os.path.join(ROOT, "outputs", "cctv_" + v, "pcam", "calib", uid + ".npz")
        sp = os.path.join(ROOT, "outputs", "cctv_" + v, os.environ.get("PCAM_SHUTTLE", "shuttle_stab08"), uid + ".json")
        if not (os.path.exists(cp) and os.path.exists(sp)):
            continue
        z = np.load(cp); S = json.load(open(sp)); sh = PCAM_SYNC.get(v, float(z["shift"]))
        R = Rotation.from_rotvec(z["rvec"]).as_matrix()
        rows = [(k - sh, np.array(pk)[:, :2], R[k], float(z["f"][k])) for k, pk in zip(S["frames"], S["peaks"]) if pk and k < len(R)]  # TV time, fractional
        out[v] = (rows, np.asarray(z["C"], float), float(z["k1"]))
    return out


MIN_TRACK = int(os.environ.get("T_MIN_TRACK", "5"))  # frames: a player-camera track must be this long to count
MAX_STEP = 1.4    # m per frame (~35 m/s)
MAX_ACC = 0.35    # m per frame^2 second difference: a flying shuttle is smooth from frame to frame


def _tracks(nodes):
    """nodes [(t, X, payload)] -> smooth 3D tracks: longest chains where each step is <= MAX_STEP per frame and the
    second difference <= MAX_ACC (gaps of 1-2 frames allowed). Greedy, longest first; each node used once."""
    nodes = sorted(nodes, key=lambda n: n[0])
    N = len(nodes)
    used = np.zeros(N, bool)
    out = []
    while True:
        best, prev = np.zeros(N, int), np.full(N, -1)
        for i in range(N):
            if used[i]:
                continue
            best[i] = 1
            ti, Xi = nodes[i][0], nodes[i][1]
            for j in range(i - 1, -1, -1):
                tj, Xj = nodes[j][0], nodes[j][1]
                if ti - tj > 3:
                    break
                if used[j] or ti == tj or np.linalg.norm(Xi - Xj) > MAX_STEP * (ti - tj):
                    continue
                k = prev[j]
                if k >= 0:  # smoothness against the chain's previous step (velocity per frame)
                    v1 = (Xj - nodes[k][1]) / (tj - nodes[k][0]); v2 = (Xi - Xj) / (ti - tj)
                    if np.linalg.norm(v2 - v1) > MAX_ACC * max(1, ti - tj):
                        continue
                if best[j] + 1 > best[i]:
                    best[i], prev[i] = best[j] + 1, j
        if not N or best.max() < MIN_TRACK:
            return out
        i = int(best.argmax()); chain = []
        while i >= 0:
            chain.append(i); i = prev[i]
        used[chain] = True
        out.append([nodes[i] for i in chain[::-1]])


STATIC_DEG = 0.4  # a peak pointing within this of a direction peaked >= STATIC_DT frames away is a fixed object
STATIC_DT = 8


def _static(rows, C, k1):
    """Per row, per peak: True where the peak's world direction recurs far apart in time -- a fixed thing in the hall
    (umpire, ad board corner), not a shuttle. A fixed point + a moving TV ray makes a smooth fake 3D track otherwise."""
    from birdseye_v2.teacher_dense import ObsCams
    T, D, owner = [], [], []
    for r, (t, pk, R, f) in enumerate(rows):
        _, d = ObsCams(np.repeat(R[None], len(pk), 0), np.full(len(pk), f), C, k1).rays(pk)
        for i in range(len(pk)):
            T.append(t); D.append(d[i]); owner.append((r, i))
    out = [np.zeros(len(r[1]), bool) for r in rows]
    if not T:
        return out
    T, D = np.array(T), np.array(D)
    cos = np.cos(np.radians(STATIC_DEG))
    for a in range(len(T)):
        far = np.abs(T - T[a]) >= STATIC_DT
        if far.any() and np.sum(D[far] @ D[a] > cos) >= 2:
            r, i = owner[a]; out[r][i] = True
    return out


SEED_M = float(os.environ.get("T_SEED_M", "0.25"))   # m: a player-camera peak this close to the two-view shuttle is a seed
TRACK_DEG = float(os.environ.get("T_TRACK_DEG", "1.2"))  # deg: angular tolerance around the predicted direction
MAX_MISS = 2


def _seeded_tracks(rows, C, k1, tri):
    """Player-camera shuttle tracks in direction space, started only from peaks that agree with the two-view (TV+side)
    shuttle and followed frame to frame by constant angular velocity -> {row index: peak index}."""
    from birdseye_v2.teacher_dense import ObsCams
    static = _static(rows, C, k1)
    T = np.array([r[0] for r in rows])
    dirs = []
    for (t, pk, R, f), st in zip(rows, static):
        _, d = ObsCams(np.repeat(R[None], len(pk), 0), np.full(len(pk), f), C, k1).rays(pk)
        d[st] = np.nan
        dirs.append(d)
    tt = tri[:, 0]
    seeds = []
    for r, (t, pk, R, f) in enumerate(rows):
        i = np.searchsorted(tt, t)
        if i == 0 or i >= len(tt) or tt[i] - tt[i - 1] > 1.01:
            continue
        a = t - tt[i - 1]; X = tri[i - 1, 1:] * (1 - a) + tri[i, 1:] * a
        u = (X - C) / np.linalg.norm(X - C); dist = np.linalg.norm(X - C)
        ang = np.arccos(np.clip(np.nan_to_num(dirs[r] @ u, nan=-1), -1, 1))
        j = int(np.argmin(ang))
        if ang[j] * dist < SEED_M:
            seeds.append((r, j))
    take = {}
    tol = np.radians(TRACK_DEG)
    for r0, j0 in seeds:
        take[r0] = j0
        for step in (1, -1):
            hist = [(T[r0], dirs[r0][j0])]
            r, miss = r0, 0
            while 0 <= r + step < len(rows) and miss <= MAX_MISS:
                r += step
                dt = T[r] - hist[-1][0]
                if abs(dt) > 1 + MAX_MISS + 0.01:
                    break
                pred = hist[-1][1] + (hist[-1][1] - hist[-2][1]) / (hist[-1][0] - hist[-2][0]) * dt if len(hist) > 1 else hist[-1][1]
                pred = pred / np.linalg.norm(pred)
                ang = np.arccos(np.clip(np.nan_to_num(dirs[r] @ pred, nan=-1), -1, 1))
                j = int(np.argmin(ang))
                if ang[j] < tol * (1 + 0.5 * miss):
                    if r in take and take[r] != j:
                        break  # ran into another track
                    take[r] = j; hist.append((T[r], dirs[r][j])); miss = 0
                else:
                    miss += 1
    return take


def add_pcams(V, tri, cand):
    """Views with the player cameras as extra (per-frame posed) views: their shuttle tracks seeded where the TV and
    side camera triangulate and followed into the stretches the side camera cannot see (_seeded_tracks). Where a track
    point's ray meets a TV ray (GAP), the meeting point joins the initial guesses. -> (V4, tri4, leftovers)."""
    from birdseye_v2.teacher import Views
    from birdseye_v2.teacher_dense import ObsCams
    det = [{int(round(f)): p for f, p in zip(V.frames[j], V.px[j])} for j in range(2)]

    def ref_ray(j, t):
        a, b = int(np.floor(t)), int(np.ceil(t))
        if a == b or abs(t - round(t)) < 0.05:
            p = det[j].get(int(round(t)))
        elif a in det[j] and b in det[j]:
            p = det[j][a] * (b - t) + det[j][b] * (t - a)
        else:
            p = None
        if p is None:
            return None
        c, d = V.cams[j].rays(np.asarray(p)[None])
        return c, d[0]
    have = {int(r[0]) for r in tri}
    cams, frames, px = list(V.cams), list(V.frames), list(V.px)
    newtri, left = [], {}
    for v, (rows, C, k1) in cand.items():
        take = _seeded_tracks(rows, C, k1, tri)
        Rs, fs, fr_, px_ = [], [], [], []
        for r in sorted(take):
            t, pk, R, f = rows[r]; i = take[r]
            Rs.append(R); fs.append(f); fr_.append(t); px_.append(pk[i])
            _, d1 = ObsCams(R[None], [f], C, k1).rays(pk[i][None])
            rr = ref_ray(0, t)
            if rr is not None and int(round(t)) not in have:
                g, mid, s1, s2 = _gap(C, d1[0], rr[0], rr[1])
                if mid is not None and g < GAP and s1 > 0.5 and s2 > 0.5:
                    newtri.append([int(round(t)), *mid]); have.add(int(round(t)))
        left[v] = ([], C, k1)  # tracks already carry everything the camera can confirm
        if fr_:
            cams.append(ObsCams(np.array(Rs), np.array(fs), C, k1)); frames.append(np.array(fr_, float)); px.append(np.array(px_))
    V4 = Views(cams, frames, px, foot=V.foot, C=V.C, rk_dir=V.rk_dir, rk_conf=V.rk_conf)
    tri4 = np.concatenate([tri, np.array(newtri).reshape(-1, 4)]) if newtri else tri
    return V4, tri4[np.argsort(tri4[:, 0])], left


def self_pass(V4, out, left):
    """Second pass: player-camera detections close to the teacher's own path (frames no other view saw)."""
    from birdseye_v2.teacher import Views
    from birdseye_v2.teacher_dense import ObsCams, positions
    cams, frames, px = list(V4.cams), list(V4.frames), list(V4.px)
    n = 0
    for v, (rest, C, k1) in left.items():
        Rs, fs, fr_, px_ = [], [], [], []
        for t, pk, R, f in rest:
            P = positions(out, [t])[0]
            if not np.isfinite(P).all():
                continue
            q = ObsCams(R[None], [f], C, k1).project(P[None])[0]
            d = np.linalg.norm(pk - q, axis=1)
            if d.min() < SELF_PX:
                Rs.append(R); fs.append(f); fr_.append(t); px_.append(pk[d.argmin()])
        if fr_:
            cams.append(ObsCams(np.array(Rs), np.array(fs), C, k1)); frames.append(np.array(fr_, float)); px.append(np.array(px_))
            n += len(fr_)
    return Views(cams, frames, px, foot=V4.foot, C=V4.C, rk_dir=V4.rk_dir, rk_conf=V4.rk_conf), n


G_SDK = float(os.environ.get("T_G_SDK", "2.5"))     # search +- this many teacher sds along the TV ray
G_PX = float(os.environ.get("T_G_PX", "6.0"))       # px: a player-camera peak this close to that segment
G_GAP = float(os.environ.get("T_G_GAP", "0.15"))    # m: and its ray this close to the TV ray
G_MIN = int(os.environ.get("T_G_MIN", "3"))         # frames: smooth track length to be kept
G_ALL = os.environ.get("T_G_ALL", "0") == "1"       # search every frame the side camera misses (not only unsure ones)
G_SEG_MIN = float(os.environ.get("T_G_SEG", "0"))   # m: search at least +- this far along the TV ray


def guided_pass(V, out, cand):
    """Player-camera detections where the teacher is unsure: in frames the TV sees, the shuttle is on the TV ray within
    the teacher's own depth interval (+- G_SDK sd) -- a short segment in each player camera. Peaks on that segment
    whose rays meet the TV ray (G_GAP), not fixed hall objects (_static), chained into smooth tracks (_tracks)
    -> (Views with them added, tri points to add, count)."""
    from birdseye_v2.teacher import Views
    from birdseye_v2.teacher_dense import ObsCams, positions, confidence
    fr = np.asarray(out["frames"]); sd = np.interp
    tvp = {int(round(f)): p for f, p in zip(V.frames[0], V.px[0])}
    side = set(np.round(V.frames[1]).astype(int).tolist()) if len(V.frames) > 1 else set()
    cams, frames, px = list(V.cams), list(V.frames), list(V.px)
    pts, n = [], 0
    for v, (rows, C, k1) in cand.items():
        static = _static(rows, C, k1)
        nodes = []
        for (t, pk, R, f), st in zip(rows, static):
            ti = int(round(t))
            if ti not in tvp or (ti in side if G_ALL else confidence(out, [t])[0]):
                continue
            P = positions(out, [t])[0]
            if not np.isfinite(P).all():
                continue
            s_ = float(np.interp(t, fr, out["sd"]))
            c, d = V.cams[0].rays(np.asarray(tvp[ti])[None]); d = d[0]
            dep = (P - c) @ d
            hw = max(G_SDK * s_, G_SEG_MIN)
            seg = c + np.linspace(max(dep - hw, 1.0), dep + hw, 80)[:, None] * d
            oc = ObsCams(np.repeat(R[None], len(seg), 0), np.full(len(seg), f), C, k1)
            q = oc.project(seg)
            z = np.einsum("ij,mj->mi", R, seg - C)[:, 2]
            q = q[z > 0.3]
            if len(q) < 2:
                continue
            _, dirs = ObsCams(np.repeat(R[None], len(pk), 0), np.full(len(pk), f), C, k1).rays(pk)
            for i in range(len(pk)):
                if st[i]:
                    continue
                if np.min(np.linalg.norm(q - pk[i], axis=1)) > G_PX:
                    continue
                g, mid, s1, s2 = _gap(C, dirs[i], c, d)
                if mid is not None and g < G_GAP and s1 > 0.5 and s2 > 0.5 and -0.05 < mid[2] < 9:
                    nodes.append((t, mid, (i, R, f, pk)))
        global MIN_TRACK
        keep, MIN_TRACK = MIN_TRACK, G_MIN
        try:
            trs = _tracks(nodes)
        finally:
            MIN_TRACK = keep
        Rs, fs, fr_, px_ = [], [], [], []
        for tr in trs:
            for t, X, (i, R, f, pk) in tr:
                Rs.append(R); fs.append(f); fr_.append(t); px_.append(pk[i]); pts.append([int(round(t)), *X])
        if fr_:
            o = np.argsort(fr_)
            cams.append(ObsCams(np.array(Rs)[o], np.array(fs)[o], C, k1)); frames.append(np.array(fr_, float)[o]); px.append(np.array(px_)[o])
            n += len(fr_)
    return Views(cams, frames, px, foot=V.foot, C=V.C, rk_dir=V.rk_dir, rk_conf=V.rk_conf), np.array(pts).reshape(-1, 4), n


def teach_all(V, tri, cand):
    """The teacher with every view: TV + side, then player-camera detections searched where it is unsure (guided_pass),
    then solved again with them."""
    from birdseye_v2 import teacher_dense as TE
    out = TE.teach(V, tri)
    if not cand:
        return out, {"pcam_guided": 0}
    Vg, pts, n = guided_pass(V, out, cand)
    if n:
        have = set(tri[:, 0].astype(int).tolist())
        add = pts[[int(p[0]) not in have for p in pts]] if len(pts) else pts
        tri2 = np.concatenate([tri, add]) if len(add) else tri
        out = TE.teach(Vg, tri2[np.argsort(tri2[:, 0])])
    return out, {"pcam_guided": n}


def one(uid):
    from birdseye_v2 import teacher_dense as TE
    dst = os.path.join(OUT, uid + ".json")
    if os.path.exists(dst):
        return uid, "done"
    try:
        V, tri, _ = load(uid)
        out, st = teach_all(V, tri, pcam_candidates(uid)) if os.environ.get("T_PCAM", "0") == "1" else (TE.teach(V, tri), {})
        out.update(st)
        fr = np.asarray(out["frames"])
        conf = TE.confidence(out, fr)
        P = np.asarray(out["P"])
        # labels: only the frames the teacher pins down (Laplace sd < CONF_SIGMA); the rest stays unlabeled
        out["S"] = [[int(f)] + [round(float(v), 4) for v in p] for f, p, c in zip(fr, P, conf) if c and np.isfinite(p).all()]
        out["conf_frac"] = round(float(conf.mean()), 3)
        out["fps"] = FPS
        json.dump(out, open(dst, "w"))
        return uid, f"{len(out['hits'])} hits, confident {conf.mean():.0%} of {len(fr)}"
    except Exception as ex:
        return uid, "FAILED " + repr(ex)[:200]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    test = set(json.load(open(os.path.join(ROOT, "outputs", "cctv", "split.json")))["test"])
    uids = sys.argv[1:] or sorted(os.path.basename(p)[:-5] for p in os.listdir(os.path.join(ROOT, "outputs", "cctv", "tri"))
                                  if p.endswith(".json"))
    uids = [u for u in uids if u not in test]
    with Pool(int(os.environ.get("NPROC", "12"))) as pool:
        for uid, msg in pool.imap_unordered(one, uids):
            print(uid, msg, flush=True)
