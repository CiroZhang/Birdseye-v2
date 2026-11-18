"""Data for the teacher's learned confidence: per-frame features and the true 3D error, from self-hiding tests.

The Laplace sd of the dense teacher is only a feature: at a wrong local minimum it is confidently wrong, and the sparse
jump cost shrinks it unevenly. So which frames become labels is learned from the rallies themselves: detections are
hidden (side camera in alternate 38-frame windows; 5-frame blocks from both cameras), the teacher runs on the rest, and
its error where the two-view triangulation exists is the target. "full" runs (nothing hidden) give the features of the
frames the labels are made from.
Output outputs/cctv/teach_calib/<uid>.npz: per mode (full, side, blocks) a table FEATS + [err, hidden, frame].
usage: python tools/teach_calib.py [N extra rallies, default 30]   (env NPROC; teacher env, e.g. T_DV_SPARSE T_GLOBAL T_PCAM)"""
import json, os, sys, zlib
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.environ.get("TC_OUT", os.path.join(ROOT, "outputs", "cctv", "teach_calib"))
PCAM = os.environ.get("T_PCAM", "0") == "1"  # teacher with the player cameras (cctv_teach.teach_all)
FEATS = ["log_sd", "log_sd_l1", "d_tv", "d_side", "d_both", "dv_max8", "n_jumps12", "z", "d_player", "speed"]


def _dist_to(fr, obs):
    """Frames from each fr to the nearest frame in obs (large if none)."""
    if len(obs) == 0:
        return np.full(len(fr), 99.0)
    obs = np.sort(np.asarray(obs, float))
    i = np.clip(np.searchsorted(obs, fr), 1, len(obs) - 1)
    return np.minimum(np.abs(fr - obs[i - 1]), np.abs(fr - obs[i])).clip(0, 99)


def features(V, out):
    fr = np.asarray(out["frames"], float)
    sd = np.asarray(out["sd"]); sdl = np.asarray(out.get("sd_l1", out["sd"]))
    dv = np.asarray(out["dv"]); P = np.asarray(out["P"]); Vv = np.asarray(out["V"])
    tv, side = V.frames[0], V.frames[1]
    both = np.intersect1d(np.round(tv), np.round(side))
    pad = lambda a, w, f: np.array([f(a[max(i - w, 0):i + w + 1]) for i in range(len(a))])
    from birdseye_v2.teacher_dense import _players
    ft = _players(V, fr)
    dp = np.linalg.norm(ft - P[:, None, :2], axis=2)
    dp = np.where(np.isfinite(dp), dp, 99).min(1)
    return np.c_[np.log(sd + 1e-4), np.log(sdl + 1e-4), _dist_to(fr, tv), _dist_to(fr, side), _dist_to(fr, both),
                 pad(dv, 8, np.max), pad(dv > 3, 12, np.sum), P[:, 2], dp.clip(0, 20), np.linalg.norm(Vv, axis=1)]


def one(uid):
    import cctv_teach as CTE
    from birdseye_v2.teacher import Views
    from birdseye_v2 import teacher_dense as TE
    dst = os.path.join(OUT, uid + ".npz")
    if os.path.exists(dst):
        return uid, "done"
    try:
        V, tri, ctx = CTE.load(uid)
        cand = CTE.pcam_candidates(uid) if PCAM else None
        res = {}
        for mode in ("full", "side", "blocks"):
            if mode == "full":
                hid = lambda f: np.zeros(len(np.atleast_1d(f)), bool)
                V2 = V
            elif mode == "side":
                hid = lambda f: (np.floor(np.asarray(f) / 38) % 2) == 1
                k = ~hid(V.frames[1])
                V2 = Views(V.cams, [V.frames[0], V.frames[1][k]], [V.px[0], V.px[1][k]], **ctx)
            else:
                rng = np.random.default_rng(zlib.crc32(uid.encode()))
                lo = min(f.min() for f in V.frames); hi = max(f.max() for f in V.frames)
                st = np.arange(lo, hi, 5)
                hb = set(st[rng.random(len(st)) < 0.25].astype(int).tolist())
                hid = lambda f, lo=lo, hb=hb: np.isin((np.floor((np.asarray(f) - lo) / 5) * 5 + lo).astype(int), list(hb))
                k0, k1 = ~hid(V.frames[0]), ~hid(V.frames[1])
                V2 = Views(V.cams, [V.frames[0][k0], V.frames[1][k1]], [V.px[0][k0], V.px[1][k1]], **ctx)
            if PCAM:  # player cameras too: only the TV-ray-guided search, against what is left after hiding
                out, st = CTE.teach_all(V2, tri[~hid(tri[:, 0])], cand)
                out.update(st)
            else:
                out = TE.teach(V2, tri[~hid(tri[:, 0])])
            fr = np.asarray(out["frames"], float)
            F = features(V2, out)
            err = np.full(len(fr), np.nan)
            idx = {int(f): i for i, f in enumerate(fr)}
            P = np.asarray(out["P"])
            for f, X in zip(tri[:, 0], tri[:, 1:]):
                if int(f) in idx:
                    err[idx[int(f)]] = np.linalg.norm(P[idx[int(f)]] - X)
            res[mode] = np.c_[F, err, hid(fr).astype(float), fr]
            if mode == "full":
                json.dump(out, open(os.path.join(OUT, uid + "_full.json"), "w"))
        np.savez_compressed(dst, **res)
        return uid, "ok"
    except Exception as ex:
        return uid, "FAILED " + repr(ex)[:200]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    test = set(json.load(open(os.path.join(ROOT, "outputs", "cctv", "split.json")))["test"])
    dev = open(os.path.join(ROOT, "outputs", "cctv_pcam_dev_uids.txt")).read().split()
    uids = sorted(p[:-5] for p in os.listdir(os.path.join(ROOT, "outputs", "cctv", "tri")) if p.endswith(".json"))
    rest = [u for u in uids if u not in test and u not in dev]
    extra = [rest[i] for i in np.random.default_rng(1).choice(len(rest), min(n, len(rest)), replace=False)]
    with Pool(int(os.environ.get("NPROC", "12"))) as pool:
        for r in pool.imap_unordered(one, [u for u in dev if u not in test] + extra):
            print(*r, flush=True)
