"""Accuracy of the two-view teacher (birdseye_v2/teacher.py) where it matters: stretches only ONE camera sees.

Per rally, the side camera's detections (and the triangulated initial guesses) are hidden in alternate windows
(HIDE frames hidden of every 2*HIDE); the teacher runs on what is left; its 3D in the hidden windows is compared with
the two-view triangulation there (1-2 cm ray gap, outputs/cctv/tri). Also: the error on frames both views see, and
the fraction of frames that are physically impossible (outside the hall box, above 6 m, below the floor).
Output: outputs/cctv/teach_eval/<tag>.json; prints the summary.
usage: python tools/teach_eval.py <tag> [N rallies, default 48]   (env NPROC, HIDE (frames, default 38), teacher env)"""
import json, os, sys, zlib
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
HIDE = int(os.environ.get("HIDE", "38"))


def impossible(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return (z > 6) | (z < -0.05) | (x < -2) | (x > 8.1) | (y < -3) | (y > 16.4)


def one(uid):
    import cctv_teach as CTE
    from birdseye_v2.teacher import Views
    import importlib
    TE = importlib.import_module("birdseye_v2." + os.environ.get("T_MODE", "teacher"))
    try:
        V, tri, ctx = CTE.load(uid)
        use_p = os.environ.get("T_PCAM") == "1"
        cand = CTE.pcam_candidates(uid) if use_p else {}
        side_f = set(tri[:, 0].astype(int).tolist())
        if use_p and os.environ.get("GT_PCAM", "1") == "1" and cand:
            tri = CTE.add_pcams(V, tri, cand)[1]  # ground truth also where TV + a player camera triangulate
        gapgt = ~np.isin(tri[:, 0].astype(int), list(side_f))  # points only a player camera gives: the side camera's gaps
        if os.environ.get("HIDE_MODE") == "blocks":
            # random 5-frame blocks (~25% of the rally) hidden from BOTH views: what the teacher must fill in
            rng = np.random.default_rng(zlib.crc32(uid.encode()))
            lo_, hi_ = min(f.min() for f in V.frames), max(f.max() for f in V.frames)
            starts = np.arange(lo_, hi_, 5)
            hb = set(starts[rng.random(len(starts)) < 0.25].astype(int).tolist())
            hid = lambda f: np.isin((np.floor((np.asarray(f) - lo_) / 5) * 5 + lo_).astype(int), list(hb))
            k0, k1 = ~hid(V.frames[0]), ~hid(V.frames[1])
            V2 = Views(V.cams, [V.frames[0][k0], V.frames[1][k1]], [V.px[0][k0], V.px[1][k1]], **ctx)
            cand = {v: ([r for r in rows if not hid(np.array([r[0]]))[0]], C, k1) for v, (rows, C, k1) in cand.items()}
        else:
            hid = lambda f: (np.floor(np.asarray(f) / HIDE) % 2) == 1
            keep = ~hid(V.frames[1])
            V2 = Views(V.cams, [V.frames[0], V.frames[1][keep]], [V.px[0], V.px[1][keep]], **ctx)
        # T_PCAM_TEACH=0: same (extended) ground truth, but the teacher without the player cameras -- the baseline
        use_t = use_p and os.environ.get("T_PCAM_TEACH", "1") == "1"
        out = CTE.teach_all(V2, tri[~hid(tri[:, 0])], cand)[0] if use_t else TE.teach(V2, tri[~hid(tri[:, 0])])
        P = TE.positions(out, tri[:, 0])
        e = np.linalg.norm(P - tri[:, 1:], axis=1)
        h = hid(tri[:, 0])
        lo, hi = out["shots"][0][0], out["shots"][-1][1]
        A = TE.positions(out, np.arange(np.ceil(lo), np.floor(hi) + 1))
        A = A[np.isfinite(A).all(1)]
        conf = TE.confidence(out, tri[:, 0]) if hasattr(TE, "confidence") else np.ones(len(tri), bool)
        g = h & gapgt
        # mid-air direction changes: velocity jumps > 2 m/s with no player within 1.5 m (horizontal) -- impossible
        fr_ = np.asarray(out["frames"]); Pp = np.asarray(out["P"]); dv = np.asarray(out["dv"])
        ft = V.foot[np.clip(fr_.astype(int), 0, len(V.foot) - 1)] if V.foot is not None else np.full((len(fr_), 4, 2), np.nan)
        dd = np.linalg.norm(ft - Pp[:, None, :2], axis=2); dd = np.where(np.isfinite(dd), dd, 99).min(1)
        air = int(((dv > 2.0) & (dd > 1.5)).sum())
        return {"uid": uid, "hidden": float(np.nanmedian(e[h])) if h.any() else None,
                "air_jumps": air, "gap_n": int(g.sum()), "gap_err": e[g].round(3).tolist(), "gap_conf": conf[g].astype(int).tolist(),
                "hidden_conf": float(np.nanmedian(e[h & conf])) if (h & conf).any() else None,
                "conf_cover": float(conf[h].mean()) if h.any() else None,
                "seen": float(np.nanmedian(e[~h])) if (~h).any() else None,
                "nan_hidden": float(np.isnan(e[h]).mean()) if h.any() else None,
                "impossible": float(impossible(A).mean()) if len(A) else None,
                "cover_all": float(TE.confidence(out, np.arange(np.ceil(lo), np.floor(hi) + 1)).mean()) if hasattr(TE, "confidence") else None,
                "sd_err": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(
                    np.interp(tri[h, 0], out["frames"], out["sd"]) if "sd" in out else np.zeros(h.sum()), e[h])],
                "flights": len(out["shots"]), "hits": len(out["hits"]), "dur": (hi - lo) / 25}
    except Exception as ex:
        return {"uid": uid, "error": repr(ex)[:200]}


def summary(R):
    R = [r for r in R if "error" not in r and r["hidden"] is not None]
    g = lambda k: np.array([r[k] for r in R if r.get(k) is not None], float)
    h = g("hidden")
    s = {"rallies": len(R), "hidden_med": round(float(np.median(h)), 3), "hidden_p75": round(float(np.percentile(h, 75)), 3),
         "hidden_p90": round(float(np.percentile(h, 90)), 3), "seen_med": round(float(np.median(g("seen"))), 3),
         "impossible": round(float(np.mean(g("impossible"))), 4),
         "hits_per_s": round(float(np.median(g("hits") / g("dur"))), 2),
         "air_jumps_per_s": round(float(np.sum(g("air_jumps")) / np.sum(g("dur"))), 2)}
    if len(g("hidden_conf")):
        s["hidden_conf_med"] = round(float(np.median(g("hidden_conf"))), 3)
        s["conf_cover"] = round(float(np.mean(g("conf_cover"))), 3)
        s["hidden_conf_p90"] = round(float(np.percentile(g("hidden_conf"), 90)), 3)
    if len(g("cover_all")):
        s["cover_all"] = round(float(np.mean(g("cover_all"))), 3)
    ge = np.concatenate([np.array(r.get("gap_err", []), float) for r in R]) if R else np.zeros(0)
    gc = np.concatenate([np.array(r.get("gap_conf", []), bool) for r in R]) if R else np.zeros(0, bool)
    if len(ge):
        s["gap_points"] = int(len(ge)); s["gap_med"] = round(float(np.nanmedian(ge)), 3)
        s["gap_p90"] = round(float(np.nanpercentile(ge, 90)), 3); s["gap_conf_cover"] = round(float(gc.mean()), 3)
        if gc.any():
            s["gap_conf_med"] = round(float(np.nanmedian(ge[gc])), 3)
    return s


if __name__ == "__main__":
    tag = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    test = set(json.load(open(os.path.join(ROOT, "outputs", "cctv", "split.json")))["test"])
    uids = sorted(p[:-5] for p in os.listdir(os.path.join(ROOT, "outputs", "cctv", "tri")) if p.endswith(".json"))
    uids = [u for u in uids if u not in test]
    uids = [uids[i] for i in np.random.default_rng(0).choice(len(uids), min(n, len(uids)), replace=False)]
    if os.environ.get("UIDS"):  # a fixed list instead (e.g. the player-camera dev rallies)
        uids = [u for u in open(os.environ["UIDS"]).read().split() if u not in test]
    with Pool(int(os.environ.get("NPROC", "12"))) as pool:
        R = []
        for r in pool.imap_unordered(one, uids):
            R.append(r)
            print(r, flush=True)
    od = os.path.join(ROOT, "outputs", "cctv", "teach_eval")
    os.makedirs(od, exist_ok=True)
    s = summary(R)
    json.dump({"summary": s, "env": {k: v for k, v in os.environ.items() if k.isupper() and len(k) < 20 and k.startswith(("T_", "HIDE", "SPLIT"))},
               "rallies": R}, open(os.path.join(od, tag + ".json"), "w"), indent=1)
    print("SUMMARY", tag, s)
