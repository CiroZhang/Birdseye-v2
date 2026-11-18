"""Evaluate back-view-only 3D shuttle reconstruction against the two-view DroneData GT.

For each rally with GT (tools/build_gt.py) + back-view pose (tools/run_pose.py):
  inputs (back view ONLY): back camera calibration, refined back-view TrackNet track, 2D pose
  hits:  'oracle'   = annotated hits (top-view clock) shifted into the back clock
         'detected' = hits detected from the back-view 2D track (shuttle3d.detect_hits)
  methods:
    v1        Birdseye v1 per-shot fit (third_party/birdseye_v1, verbatim) with hitter/receiver
              hand anchors from the same lifted pose
    v2        joint rally fit (birdseye_v2.shuttle3d.RallyFit)
    v2-nocont / v2-nonet / v2-nohand / v2-l2   ablations
  error: at every GT point (top frame f -> back time f+offset), Euclidean / ground-plane /
         height error in metres.

Writes outputs/eval/<tag>.json (per-rally + pooled) and prints the summary table.
usage: python tools/eval_shuttle.py [--hits oracle|detected] [--tag name] [BD12 ...]"""
import argparse, csv, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))
from birdseye_v1 import shuttle_physics as v1
from birdseye_v2.court import NET_Y, POST_H, Camera
from birdseye_v2.footlock import stabilize
from birdseye_v2.players import load_pose, load_rackets, players_3d, racket_anchor
from birdseye_v2.twoview import kink_time
from birdseye_v2.hitclf import detect_hits_learned
from birdseye_v2.shuttle3d import (FPS, RallyFit, clean_track, detect_hits, fit_rally, hand_candidates,
                                   landing_frame, merge_explained, near_player_px, segment_shots,
                                   split_unexplained)

GT = os.path.join(ROOT, "outputs", "gt")
XTOP = os.environ.get("XTOP", "0") == "1"  # score against the top view's own detections (tools/xview_eval.py)
HITCLF = None  # set by --hitclf (learned hit detector); module-level so worker processes see it
POSE = os.path.join(ROOT, "outputs", "pose")
CALIB = os.path.join(ROOT, "outputs", os.environ.get("DRONE_CALIB", "calib"))  # back camera only is read
CSV_DIR = os.path.join(ROOT, "data", "datacsv")
OUT = os.path.join(ROOT, "outputs", "eval")
ANCHOR = os.environ.get("ANCHOR", "yolo")  # "vimo": hand anchors from TRAM/VIMO bodies (tools/drone_vimo.py)
VIMO_J = None  # (P,T,24,3) world joints of the rally being evaluated, when ANCHOR=vimo
RACKET_K = float(os.environ.get("RACKET_K", "0"))  # ANCHOR=vimo: anchor this far (m) past the hand along the forearm
# (racket head 0.48: better on BD15-18 -- detected mean 1.01 -> 0.97 m -- but not on BD19-23, 1.10 -> 1.13; off)


def reach(J, hand, wrist, elbow, k=RACKET_K, min_rise=0.2):
    """Racket-head anchor: k m past the SMPL hand along the forearm (head held up >= min_rise, as the viewer)."""
    if k <= 0:
        return J[hand]
    u = J[wrist] - J[elbow]
    u = u / (np.linalg.norm(u) + 1e-9)
    if u[2] < min_rise:
        h = np.linalg.norm(u[:2]) + 1e-9
        u = np.r_[u[:2] / h * np.sqrt(1 - min_rise ** 2), min_rise]
    return J[hand] + k * u


def interp_px(frames, px, t):
    if len(frames) == 0:
        return None
    i = np.searchsorted(frames, t)
    if i <= 0:
        return px[0]
    if i >= len(frames):
        return px[-1]
    a = (t - frames[i - 1]) / max(frames[i] - frames[i - 1], 1e-9)
    return px[i - 1] * (1 - a) + px[i] * a


def hands_for(cam, poses, frames, px, t):
    """3D hand of the player closest (in the image) to the shuttle at back-clock time t."""
    c = hand_cands_for(cam, poses, frames, px, t, max_px=1e9)
    return c[0]["hand"] if c else None


def hand_cands_for(cam, poses, frames, px, t, max_px=300.0, rackets=None):
    f = int(round(t))
    if f < 0 or f >= len(poses):
        return []
    uv = interp_px(frames, px, t)
    if uv is None:
        return []
    people = players_3d(cam, poses[f])
    if VIMO_J is not None:  # same candidates, but each player's hands from the VIMO body
        people = []
        for J in (VIMO_J[:, f] if f < VIMO_J.shape[1] else []):
            if np.isfinite(J[[22, 23, 7, 8]]).all():
                people.append({"wrists": [reach(J, 22, 20, 18), reach(J, 23, 21, 19)], "foot": J[[7, 8], :2].mean(0),
                               "kp": np.c_[cam.project(J), np.ones(len(J))]})  # projected joints (racket reach box)
    cands = hand_candidates(people, uv, cam, max_px)
    if rackets is not None and f < len(rackets):
        # swap each player's wrist for their racket head when a racket box is detected
        for c in cands:
            for p in people:
                if any(np.allclose(c["hand"], w) for w in p["wrists"]):
                    ra = racket_anchor(cam, p, rackets[f], uv)
                    if ra is not None:
                        c["hand"] = ra
                    break
    return cands


def fit_v1(cam, frames, px, shots, hands):
    """Birdseye v1: each shot independently, start anchor = hitter hand, end anchor = next hitter's
    hand (none for the final shot)."""
    K, R, t = cam.K, cam.R, cam.tvec
    out = []
    for i, (a, b) in enumerate(shots):
        m = (frames >= a) & (frames <= b)
        if m.sum() < 3 or hands[i] is None:
            out.append(None)
            continue
        end = hands[i + 1] if i + 1 < len(shots) else None
        try:
            p0, v0, _, valid = v1.fit_shot((frames[m] - a).astype(float), px[m], FPS, K, R, t, hands[i], 0.0,
                                           end, (b - a) if end is not None else None)
        except Exception:
            valid = False
        out.append((p0, v0) if valid else None)

    def position(tq):
        tq = np.asarray(tq, float)
        res = np.full((len(tq), 3), np.nan)
        for i, (a, b) in enumerate(shots):
            if out[i] is None:
                continue
            msk = (tq >= a) & (tq <= b)
            if msk.any():
                ts = (tq[msk] - a) / FPS
                o = np.argsort(ts)
                try:
                    sim = v1.simulate(out[i][0], out[i][1], np.r_[0.0, ts[o]])[1:]
                except Exception:
                    continue
                tmp = np.empty_like(sim)
                tmp[o] = sim
                res[msk] = tmp
        return res
    return position


METHODS = {
    "v2": {"use_net": False},
    "v2-net": {},
    "v2-land": {"use_net": False, "land": True},
    "v2-land-net": {"land": True},  # = birdseye_v2.pipeline
    "v2-ln-h03": {"land": True, "sigma_hand": 0.3},
    "v2-ln-h015": {"land": True, "sigma_hand": 0.15},
    "v2-kink": {"use_net": False, "kink": True},
    "v2-racket": {"use_net": False, "racket": True},
    "v2-h03": {"use_net": False, "sigma_hand": 0.3},
    "v2-h10": {"use_net": False, "sigma_hand": 1.0},
    "v2-nocont": {"use_continuity": False, "use_net": False},
    "v2-l2": {"robust": False, "use_net": False},
    "v2-nohand": {"nohand": True, "use_net": False},
    "v2-nearesthand": {"nearest": True, "use_net": False},
}


def eval_rally(m, r, hit_mode, only=None):
    g = json.load(open(os.path.join(GT, m, r + ".json")))
    if not g["sync"] or not g["points"]:
        return None
    off = g["sync"]["offset"]
    cam = Camera.from_dict(json.load(open(os.path.join(CALIB, m, r + ".json")))["back"])
    poses = load_pose(os.path.join(POSE, m, r + ".npz"))
    global VIMO_J
    VIMO_J = None
    if ANCHOR == "vimo":
        vp = os.path.join(ROOT, "outputs", os.environ.get("DRONE_VIMO", "drone_vimo"), m, r + ".npz")
        if not os.path.exists(vp):
            return None  # compare on rallies that have VIMO only
        VIMO_J = np.load(vp)["joints"]
        if os.environ.get("STABILIZE", "1") == "1":
            VIMO_J = np.stack([stabilize(J) for J in VIMO_J])
    frames = np.array(g["back_track"]["frames"], float)
    px = np.array(g["back_track"]["px"], float).reshape(-1, 2)
    if len(frames) < 10:
        return None
    gt_hits = [int(x["frame"]) + off for x in csv.DictReader(open(os.path.join(CSV_DIR, m, r, "1.csv")))
               if x["status"] not in ("0", "")]
    seg_info = None
    if hit_mode == "oracle":
        hits = list(gt_hits)
    elif hit_mode == "joint":  # hits chosen jointly with the flights (birdseye_v2.rallyseg)
        from birdseye_v2 import rallyseg
        lf = landing_frame(frames, px)
        frames, px = clean_track(frames, px, [lf - 1] if lf is not None else [frames[-1]])
        seg_info = rallyseg.segment(cam, frames, px, lambda t: hand_cands_for(cam, poses, frames, px, t, max_px=1e9),
                                    lam=float(os.environ.get("SEG_LAMBDA", rallyseg.LAMBDA)))
        hits = [float(frames[0])] + list(seg_info["hits"])
    else:  # fully automatic: everything from the back view
        lf = landing_frame(frames, px)
        f_c, p_c = clean_track(frames, px, [lf - 1] if lf is not None else [frames[-1]])
        if HITCLF is not None:
            hits = detect_hits_learned(HITCLF["model"], HITCLF["threshold"], cam, f_c, p_c, poses)
        else:
            hits = list(detect_hits(f_c, p_c, cam=cam, poses=poses))
        if lf is not None:
            hits = [h for h in hits if h < lf - 2]
        if not hits or hits[0] > f_c[0] + 4:  # flight already under way when the track starts
            hits = [float(f_c[0])] + hits
    frames, px = clean_track(frames, px, hits)  # same cleaned input for every method
    shots = segment_shots(frames, hits, px)
    if hit_mode == "detected":  # recover hits hidden from the detector: split non-ballistic shots
        seg = float(os.environ.get("SEG_RMS", 16.0))  # px; dev sweep 4/6/9/12/16/24 -> plateau from 12
        if os.environ.get("SEG_OFF") != "1":
            shots = merge_explained(cam, frames, px, split_unexplained(cam, frames, px, shots, max_rms=seg), max_rms=seg)
    # hit times refined to the velocity kink of the back-view track (sub-frame)
    kh = []
    for h in hits:
        k = kink_time(frames, px, h, win=5)
        kh.append(k if k is not None and abs(k - h) <= 3 else h)
    shots_k = segment_shots(frames, kh, px)
    cands_k = [hand_cands_for(cam, poses, frames, px, a, max_px=1e9) for a, _ in shots_k]
    hands = [hands_for(cam, poses, frames, px, a) for a, _ in shots]  # v1: nearest in the image
    cands = [hand_cands_for(cam, poses, frames, px, a, max_px=1e9) for a, _ in shots]
    rk_path = os.path.join(ROOT, "outputs", "racket", m, r + ".npz")
    rackets = load_rackets(rk_path) if os.path.exists(rk_path) else None
    cands_r = ([hand_cands_for(cam, poses, frames, px, a, max_px=1e9, rackets=rackets) for a, _ in shots]
               if rackets is not None else None)
    if hit_mode in ("detected", "joint"):  # shot starts not at a player (mid-flight track start, split) get no hand
        for i, (a, _) in enumerate(shots):
            f = int(round(a))
            uv = interp_px(frames, px, a)
            if not (0 <= f < len(poses)) or uv is None or near_player_px(poses[f], uv) > 80:
                cands[i] = []

    gt_t = np.array([p["frame"] for p in g["points"]], float) + off
    gt_X = np.array([p["X"] for p in g["points"]])
    if XTOP:  # cross-view check: every top-view shuttle detection, scored as distance to its top-camera ray
        top = Camera.from_dict(json.load(open(os.path.join(CALIB, m, r + ".json")))["top"])
        tt = g["top_track"]
        gt_t = np.array(tt["frames"], float) + off
        top_c, top_d = top.rays(np.array(tt["px"], float).reshape(-1, 2))
        gt_X = np.zeros((len(gt_t), 3))
    preds, net = {}, {}  # net: (crossings, crossings under the tape) of the fitted flight
    t0 = time.time()
    if only is None or "v1" in only:
        preds["v1"] = fit_v1(cam, frames, px, shots, hands)(gt_t)
    if only is not None and "mt" in only:  # MonoTrack's per-shot fit (birdseye_v2/monotrack_fit.py), same inputs
        from birdseye_v2.monotrack_fit import reconstruct as mt_reconstruct
        from birdseye_v2.players import foot_point

        def foot_at(f):  # the player nearest the shuttle at frame f (MonoTrack: the hitter's / receiver's base)
            f = int(round(f))
            uv = interp_px(frames, px, f)
            if not (0 <= f < len(poses)) or uv is None:
                return None
            best, bd = None, 1e9
            for kp in poses[f]:
                k = kp[kp[:, 2] > 0.3]
                if len(k) >= 3 and np.linalg.norm(k[:, :2].mean(0) - uv) < bd:
                    bd, best = np.linalg.norm(k[:, :2].mean(0) - uv), kp
            return None if best is None else foot_point(cam, best)
        mt_hits = [int(round(a)) for a, _ in shots] + [int(round(shots[-1][1]))]
        preds["mt"] = mt_reconstruct(cam, frames, px, mt_hits, foot_at, FPS)(gt_t)
    if only is not None and "ft" in only:  # FreeTrack's plane back-projection (birdseye_v2/freetrack_fit.py), same hits
        from birdseye_v2 import freetrack_fit as FT
        ft_hits = [int(round(a)) for a, _ in shots] + [int(round(shots[-1][1]))]
        preds["ft"] = FT.reconstruct(cam, frames, px, ft_hits, poses)(gt_t)
    for name, kw in METHODS.items():
        if only is not None and name not in only:
            continue
        kw = dict(kw)
        sh, cd = (shots_k, cands_k) if kw.pop("kink", False) else (shots, cands)
        if kw.pop("racket", False):
            if cands_r is None:
                continue
            cd = cands_r
        try:
            if kw.pop("nohand", False):
                f = RallyFit(cam, frames, px, shots, [None] * len(shots), **kw).fit()
            elif kw.pop("nearest", False):
                f = RallyFit(cam, frames, px, shots, hands, **kw).fit()
            else:
                f = fit_rally(cam, frames, px, sh, cd, **kw)
            preds[name] = f.position(gt_t, rest_until=frames[-1])
            ts = np.arange(f.shots[0][0], f.shots[-1][1], 0.25)
            Xd = f.position(ts)
            Xd = Xd[np.isfinite(Xd[:, 0])]
            dy = Xd[:, 1] - NET_Y
            k = np.flatnonzero(dy[:-1] * dy[1:] < 0)
            zc = Xd[k, 2] + (Xd[k + 1, 2] - Xd[k, 2]) * dy[k] / (dy[k] - dy[k + 1])
            net[name] = (int(len(zc)), int((zc < POST_H).sum()), int((zc < POST_H - 0.15).sum()))
        except Exception as e:
            print("fit failed", m, r, name, repr(e), flush=True)
            preds[name] = np.full_like(gt_X, np.nan)
    res = {"match": m, "rally": r, "n_gt": len(gt_X), "n_shots": len(shots), "sec": time.time() - t0, "methods": {},
           "hits_pred": [float(h) for h in hits], "hits_gt": gt_hits, "track": [float(frames[0]), float(frames[-1])],
           "seg": {k: v for k, v in (seg_info or {}).items() if k in ("n_cands", "n_fits", "cost")}}
    for name, P in preds.items():
        ok = ~np.isnan(P[:, 0])
        if XTOP:
            v = P[ok] - top_c
            e = np.linalg.norm(v - (v * top_d[ok]).sum(1, keepdims=True) * top_d[ok], axis=1)
        else:
            e = np.linalg.norm(P[ok] - gt_X[ok], axis=1)
        full = np.full(len(gt_X), np.nan)
        full[ok] = e
        res["methods"][name] = {"coverage": float(ok.mean()), "err": e.tolist(),
                                "err_full": [None if np.isnan(x) else float(x) for x in full],
                                "err_xy": np.linalg.norm(P[ok, :2] - gt_X[ok, :2], axis=1).tolist(),
                                "err_z": np.abs(P[ok, 2] - gt_X[ok, 2]).tolist(), "net": net.get(name, (0, 0, 0))}
    return res


def summarize(results):
    names = list(dict.fromkeys(n for r in results for n in r["methods"]))
    # compare on the rallies where EVERY method ran (e.g. racket boxes not yet computed)
    results = [r for r in results if all(n in r["methods"] for n in names)]
    print(f"rallies compared: {len(results)}")
    print(f"{'method':12s} {'cov':>5s} {'med3d':>6s} {'mean3d':>6s} {'medxy':>6s} {'medz':>6s} {'<0.5m':>6s} {'<1m':>6s}")
    table = {}
    for n in names:
        e = np.concatenate([r["methods"][n]["err"] for r in results])
        exy = np.concatenate([r["methods"][n]["err_xy"] for r in results])
        ez = np.concatenate([r["methods"][n]["err_z"] for r in results])
        cov = np.average([r["methods"][n]["coverage"] for r in results], weights=[r["n_gt"] for r in results])
        table[n] = {"coverage": cov, "median": float(np.median(e)), "mean": float(e.mean()),
                    "median_xy": float(np.median(exy)), "median_z": float(np.median(ez)),
                    "pct_0.5": float((e < 0.5).mean()), "pct_1.0": float((e < 1.0).mean()), "n": int(len(e))}
        nc = np.sum([(list(r["methods"][n].get("net", (0, 0))) + [0])[:3] for r in results], 0)
        table[n]["net_crossings"], table[n]["net_under"], table[n]["net_under_15cm"] = int(nc[0]), int(nc[1]), int(nc[2])
        t = table[n]
        print(f"{n:12s} {cov:5.2f} {t['median']:6.2f} {t['mean']:6.2f} {t['median_xy']:6.2f} {t['median_z']:6.2f} "
              f"{t['pct_0.5']:6.2f} {t['pct_1.0']:6.2f}  under net {t['net_under']}/{t['net_crossings']} (>15 cm under: {t['net_under_15cm']})")
    if "v1" in names:  # apples to apples: only the points v1 produced an estimate for
        print("-- on points covered by v1 --")
        for n in names:
            e = np.concatenate([[b for a, b in zip(r["methods"]["v1"]["err_full"], r["methods"][n]["err_full"])
                                 if a is not None and b is not None] for r in results])
            table[n]["same_points"] = {"median": float(np.median(e)), "mean": float(e.mean()),
                                       "pct_0.5": float((e < 0.5).mean()), "pct_1.0": float((e < 1.0).mean()),
                                       "n": int(len(e))}
            t = table[n]["same_points"]
            print(f"{n:12s} n={t['n']:6d} {t['median']:6.2f} {t['mean']:6.2f} {t['pct_0.5']:6.2f} {t['pct_1.0']:6.2f}")
    return table


def _job(args):
    m, r, hits, only, use_clf = args
    global HITCLF
    if use_clf and HITCLF is None:
        import joblib
        HITCLF = joblib.load(os.path.join(ROOT, "outputs", "hitclf.joblib"))
    try:
        return eval_rally(m, r, hits, only)
    except Exception as e:
        print("FAILED", m, r, repr(e), flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", default="oracle", choices=["oracle", "detected", "joint"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--methods", default=None, help="comma list, e.g. v1,v2")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--hitclf", action="store_true", help="detected mode: learned hit detector")
    ap.add_argument("matches", nargs="*")
    a = ap.parse_args()
    only = a.methods.split(",") if a.methods else None
    matches = a.matches or sorted(os.listdir(GT))
    jobs = []
    for m in matches:
        for gj in sorted(glob.glob(os.path.join(GT, m, "Rally-*.json"))):
            r = os.path.basename(gj)[:-5]
            if os.path.exists(os.path.join(POSE, m, r + ".npz")):
                jobs.append((m, r, a.hits, only, a.hitclf))
    results = []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(_job, jobs):
            if res:
                results.append(res)
                print(res["match"], res["rally"], "shots", res["n_shots"], "gt", res["n_gt"], " ".join(
                    f"{k}={np.median(v['err']) if v['err'] else float('nan'):.2f}" for k, v in res["methods"].items()),
                    flush=True)
    if not results:
        print("no rallies")
        return
    table = summarize(results)
    os.makedirs(OUT, exist_ok=True)
    tag = a.tag or f"{a.hits}"
    json.dump({"hits": a.hits, "table": table, "rallies": results}, open(os.path.join(OUT, tag + ".json"), "w"))


if __name__ == "__main__":
    main()
