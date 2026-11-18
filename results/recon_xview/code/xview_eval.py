"""Cross-view check on the held-out CCTV match: every method runs on the TV view only; its 3D shuttle and players are
scored in the fixed side camera, which no method sees at test time.

Error at a frame = distance (m) from the predicted 3D point to the side camera's ray through its own detection
(side frame = TV frame + the session's shift; side camera in the TV world frame, tools/cctv_tri.side_camera):
  shuttle  the refined side TrackNet track (outputs/cctv_side/refined); REF=tri keeps only side detections whose ray
           meets a TV detection's ray within 10 cm (outputs/cctv/tri), dropping side false positives (default)
  player   mid-ankle: side YOLOv8-pose (outputs/cctv_side/pose_side) people whose ankles cast onto the court (+-0.3 m;
           singles and doubles), matched per frame to the method's players by ray distance (Hungarian, <= 2.5 m)
Per rally the median over frames; table = median over rallies with a 95% bootstrap interval. Coverage = share of the
reference frames a method gives a position for.

methods  mt / v1 / v2fit  MonoTrack / Birdseye v1 / per-rally physics fit (as tools/cctv_baselines.py), shuttle
         tram  VIMO (TRAM) bodies, TRAM's own placement (outputs/cctv/vimo_bc *_raw), players
         v1p   same VIMO bodies, Birdseye v1's ankle ground correction (vimo_bc top/bottom), players
         <TAG> Birdseye v2 student(s) (STUDENTS=t3,...; each with its env), shuttle + players
out  outputs/xview/cctv_<method>.json {uid: {shuttle, shuttle_cov, player, player_cov, n_sh, n_pl}}
usage: python tools/xview_eval.py [baselines|student] [n_workers]"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
from birdseye_v2.court import L, W, NET_Y, Camera  # noqa: E402
from birdseye_v2.players import load_pose, ANKLE_H  # noqa: E402
import cctv_tri as CT  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "xview")
REF = os.environ.get("REF", "tri")
FPS = 25.0
SMPL_ANK = (7, 8)
COCO_ANK = (15, 16)
MAX_MATCH = 2.5  # m, as tools/eval_players.py: a farther prediction is not this player


def ray_dist(P, c, d):
    """Distance of points P (n,3) to rays c + s d (c (3,) or (n,3), d unit (n,3))."""
    v = P - c
    return np.linalg.norm(v - (v * d).sum(1, keepdims=True) * d, axis=1)


def side_refs(uid):
    """(side cam, shift, shuttle {tv frame: side px}, players {tv frame: {"top"/"bottom": side mid-ankle px}})."""
    sc = CT.side_camera(uid)
    if sc is None:
        return None
    side, shift = sc
    sh = CT.track(os.path.join(ROOT, "outputs", "cctv_side", "refined", uid + ".json"))
    sh = {k - shift: v for k, v in sh.items()}
    if REF == "tri":
        keep = {int(r[0]) for r in json.load(open(os.path.join(CT.OUT, uid + ".json")))["shuttle"]}
        sh = {k: v for k, v in sh.items() if k in keep}
    pl = {}
    pp = os.path.join(ROOT, "outputs", "cctv_side", "pose_side", uid + ".npz")
    if os.path.exists(pp):
        for fs, people in enumerate(load_pose(pp)):
            on = []
            for kp in people:
                a = kp[list(COCO_ANK)]
                if (a[:, 2] < 0.3).any():
                    continue
                uv = a[:, :2].mean(0)
                g = side.pixel_to_ground(uv[None], z=ANKLE_H)[0]
                if -0.3 < g[0] < W + 0.3 and -0.3 < g[1] < L + 0.3:
                    on.append(uv)
            if on:
                pl[fs - shift] = np.array(on)
    return side, shift, sh, pl


def score(side, sh, pl, shuttle=None, players=None):
    """shuttle: f(times) -> (n,3) or None; players: {side: (T,3) mid-ankle, NaN missing} or None."""
    out = {}
    if shuttle is not None and sh:
        fr = np.array(sorted(sh))
        P = shuttle(fr.astype(float))
        c, d = side.rays(np.array([sh[k] for k in fr]))
        e = ray_dist(P, np.broadcast_to(c, d.shape), d)
        ok = np.isfinite(e)
        out.update(shuttle=float(np.median(e[ok])) if ok.any() else None, shuttle_cov=float(ok.mean()), n_sh=int(len(e)))
    if players is not None and pl:  # players: (T, n, 3) mid-ankles of every predicted player (NaN missing)
        from scipy.optimize import linear_sum_assignment
        E, n = [], 0
        for f, uv in pl.items():
            n += len(uv)
            if not 0 <= f < len(players):
                continue
            pr = players[f][np.isfinite(players[f]).all(1)]
            if not len(pr):
                continue
            c, d = side.rays(uv)
            C = np.stack([ray_dist(pr, np.broadcast_to(c, (len(pr), 3)), np.broadcast_to(d[i], (len(pr), 3))) for i in range(len(uv))])
            ri, ci = linear_sum_assignment(C)
            E += [float(C[i, j]) for i, j in zip(ri, ci) if C[i, j] <= MAX_MATCH]
        if n:
            out.update(player=float(np.median(E)) if E else None, player_cov=len(E) / n, n_pl=n)
    return out


def baselines_one(uid):
    try:
        from birdseye_v2 import shuttle3d as S3
        S3.FPS = FPS
        import eval_shuttle as ES
        ES.FPS = FPS
        from birdseye_v2.players import foot_point
        from birdseye_v2.monotrack_fit import reconstruct as mt_reconstruct
        refs = side_refs(uid)
        if refs is None:
            return uid, {"error": "no side camera"}
        side, shift, sh, pl = refs
        res = {}
        vp = os.path.join(ROOT, "outputs", "cctv", "vimo_n4", uid + ".npz")
        if os.path.exists(vp):
            v = np.load(vp)
            T = int(v["frames"].max()) + 1
            for name, suf in (("tram", "_raw"), ("v1p", "")):
                A = np.full((T, 4, 3), np.nan)
                for q, h in enumerate(("top0", "top1", "bottom0", "bottom1")):
                    if h + suf in v.files:  # a side with fewer tracked people has no array for the missing one
                        A[v["frames"], q] = v[h + suf][:, list(SMPL_ANK)].mean(1)
                res[name] = score(side, sh, pl, players=A)
        cam = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "cctv", "calib_bc", uid + ".json")))["cam"])
        tr = CT.track(os.path.join(ROOT, "outputs", "cctv", "refined", uid + ".json"))
        if len(tr) < 10:
            return uid, res
        frames = np.array(sorted(tr), float)
        px = np.array([tr[k] for k in sorted(tr)], float)
        poses = load_pose(os.path.join(ROOT, "outputs", "cctv", "pose1280", uid + ".npz"))
        lf = S3.landing_frame(frames, px)
        f_c, p_c = S3.clean_track(frames, px, [lf - 1] if lf is not None else [frames[-1]])
        hits = list(S3.detect_hits(f_c, p_c, cam=cam, poses=poses))
        if lf is not None:
            hits = [h for h in hits if h < lf - 2]
        if not hits or hits[0] > f_c[0] + 4:
            hits = [float(f_c[0])] + hits
        frames, px = S3.clean_track(frames, px, hits)
        shots = S3.segment_shots(frames, hits, px)
        shots = S3.merge_explained(cam, frames, px, S3.split_unexplained(cam, frames, px, shots, max_rms=16.0), max_rms=16.0)
        hands = [ES.hands_for(cam, poses, frames, px, a) for a, _ in shots]
        cands = [ES.hand_cands_for(cam, poses, frames, px, a, max_px=1e9) for a, _ in shots]
        for i, (a, _) in enumerate(shots):
            f = int(round(a))
            uv = ES.interp_px(frames, px, a)
            if not (0 <= f < len(poses)) or uv is None or S3.near_player_px(poses[f], uv) > 80:
                cands[i] = []

        def foot_at(f):
            f = int(round(f))
            uv = ES.interp_px(frames, px, f)
            if not (0 <= f < len(poses)) or uv is None:
                return None
            best, bd = None, 1e9
            for kp in poses[f]:
                k = kp[kp[:, 2] > 0.3]
                if len(k) >= 3 and np.linalg.norm(k[:, :2].mean(0) - uv) < bd:
                    bd, best = np.linalg.norm(k[:, :2].mean(0) - uv), kp
            return None if best is None else foot_point(cam, best)

        res["v1"] = score(side, sh, pl, shuttle=ES.fit_v1(cam, frames, px, shots, hands))
        res["mt"] = score(side, sh, pl, shuttle=mt_reconstruct(cam, frames, px, [int(round(a)) for a, _ in shots] + [int(round(shots[-1][1]))], foot_at, FPS))
        try:
            fit = S3.fit_rally(cam, frames, px, shots, cands, use_net=False)
            res["v2fit"] = score(side, sh, pl, shuttle=lambda t: fit.position(t, rest_until=frames[-1]))
        except Exception:
            res["v2fit"] = score(side, sh, pl, shuttle=lambda t: np.full((len(t), 3), np.nan))
        return uid, res
    except Exception as ex:
        return uid, {"error": repr(ex)[:300]}


def ft_one(uid):
    """FreeTrack (birdseye_v2/freetrack_fit.py) with the same track, people and hits as MonoTrack in baselines_one."""
    try:
        from birdseye_v2 import shuttle3d as S3
        from birdseye_v2 import freetrack_fit as FT
        S3.FPS = FPS
        refs = side_refs(uid)
        if refs is None:
            return uid, {"error": "no side camera"}
        side, shift, sh, pl = refs
        cam = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "cctv", "calib_bc", uid + ".json")))["cam"])
        tr = CT.track(os.path.join(ROOT, "outputs", "cctv", "refined", uid + ".json"))
        if len(tr) < 10:
            return uid, {}
        frames = np.array(sorted(tr), float)
        px = np.array([tr[k] for k in sorted(tr)], float)
        poses = load_pose(os.path.join(ROOT, "outputs", "cctv", "pose1280", uid + ".npz"))
        lf = S3.landing_frame(frames, px)
        f_c, p_c = S3.clean_track(frames, px, [lf - 1] if lf is not None else [frames[-1]])
        hits = list(S3.detect_hits(f_c, p_c, cam=cam, poses=poses))
        if lf is not None:
            hits = [h for h in hits if h < lf - 2]
        if not hits or hits[0] > f_c[0] + 4:
            hits = [float(f_c[0])] + hits
        frames, px = S3.clean_track(frames, px, hits)
        shots = S3.segment_shots(frames, hits, px)
        shots = S3.merge_explained(cam, frames, px, S3.split_unexplained(cam, frames, px, shots, max_rms=16.0), max_rms=16.0)
        bounds = [int(round(a)) for a, _ in shots] + [int(round(shots[-1][1]))]
        return uid, {"ft": score(side, sh, pl, shuttle=FT.reconstruct(cam, frames, px, bounds, poses))}
    except Exception as ex:
        return uid, {"error": repr(ex)[:300]}


def test_uids():
    test = set(json.load(open(os.path.join(ROOT, "outputs", "cctv", "split.json")))["test"])
    return sorted(u for u in (p[:-5] for p in os.listdir(CT.OUT) if p.endswith(".json")) if u in test)


def dump(R, methods):
    os.makedirs(OUT, exist_ok=True)
    for m in methods:
        d = {u: v[m] for u, v in R.items() if m in v}
        json.dump(d, open(os.path.join(OUT, f"cctv_{m}.json"), "w"))
        for k in ("shuttle", "player"):
            s = [v[k] for v in d.values() if v.get(k) is not None]
            if s:
                print(f"{m:8s} {k:8s} {len(s)} rallies, median {np.median(s):.3f} m, coverage {np.mean([v[k + '_cov'] for v in d.values() if k + '_cov' in v]):.2f}", flush=True)


def student(tag):
    import train_rallynet as TR
    model = TR.load_model(tag)
    R = {}
    for k, uid in enumerate(test_uids()):
        refs = side_refs(uid)
        p = os.path.join(TR.OBS, "cctv", uid + ".npz")
        if refs is None or not os.path.exists(p):
            continue
        side, shift, sh, pl = refs
        f = TR.load_feats(p)
        o = TR.infer(model, f)
        S = o["shuttle"]
        J = o["joints24"]
        pres = f["present"] > 0
        pls = np.where(pres[..., None], J[:, :, list(SMPL_ANK)].mean(2), np.nan)  # (T, slots, 3)

        def shuttle(t, S=S):
            i = np.round(t).astype(int)
            P = np.full((len(t), 3), np.nan)
            ok = (i >= 0) & (i < len(S))
            P[ok] = S[i[ok]]
            return P
        R[uid] = score(side, sh, pl, shuttle=shuttle, players=pls)
        if k % 50 == 0:
            print(k, uid, R[uid], flush=True)
    dump({u: {tag: v} for u, v in R.items()}, [tag])


DRONE_TEST = tuple(f"BD{i}" for i in range(15, 24))


def drone_refs(m, r):
    """(top cam, sync offset, shuttle (t_back, top px), players {back frame: (n,2) court xy from the top view})."""
    from eval_players import gt_players
    g = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))
    if not g.get("sync") or not g.get("top_track"):
        return None
    top = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "calib_bc", m, r + ".json")))["top"])
    off = g["sync"]["offset"]
    t = np.array(g["top_track"]["frames"], float) + off
    px = np.array(g["top_track"]["px"], float).reshape(-1, 2)
    pl = {int(round(ft + off)): P for ft, P in gt_players(m, r, top).items()}
    return top, off, (t, px), pl


def drone_score(top, sh, pl, shuttle=None, feet=None):
    """shuttle: (T,3) back-clock frames; feet: (T, n, 2) court xy per player slot (NaN missing)."""
    from eval_players import match_err
    out = {}
    if shuttle is not None:
        t, px = sh
        k = np.floor(t).astype(int)
        ok = (k >= 0) & (k < len(shuttle) - 1)
        P = np.full((len(t), 3), np.nan)
        a = t[ok] - k[ok]
        P[ok] = shuttle[k[ok]] + a[:, None] * (shuttle[k[ok] + 1] - shuttle[k[ok]])
        c, d = top.rays(px)
        e = ray_dist(P, np.broadcast_to(c, d.shape), d)
        g = np.isfinite(e)
        out.update(shuttle=float(np.median(e[g])) if g.any() else None, shuttle_cov=float(g.mean()), n_sh=int(len(e)))
    if feet is not None:
        E, n = [], 0
        for f, G in pl.items():
            n += len(G)
            if 0 <= f < len(feet):
                pr = feet[f][np.isfinite(feet[f]).all(1)]
                E += [x for x in match_err(np.asarray(G), pr) if x is not None]
        if n:
            out.update(player=float(np.median(E)) if E else None, player_cov=len(E) / n, n_pl=n)
    return out


def drone_rallies():
    import glob
    return sorted((m, os.path.basename(p)[:-5]) for m in DRONE_TEST for p in glob.glob(os.path.join(ROOT, "outputs", "gt", m, "Rally-*.json")))


def drone_tram():
    R = {"tram": {}, "v1p": {}}
    for m, r in drone_rallies():
        refs = drone_refs(m, r)
        vp = os.path.join(ROOT, "outputs", "drone_vimo_bc", m, r + ".npz")
        if refs is None or not os.path.exists(vp):
            continue
        top, off, sh, pl = refs
        v = np.load(vp)
        for name, key in (("tram", "raw"), ("v1p", "joints")):
            feet = v[key][:, :, list(SMPL_ANK), :2].mean(2).transpose(1, 0, 2)  # (T, n, 2)
            R[name][f"{m}__{r}"] = drone_score(top, sh, pl, feet=feet)
    dump_drone(R)


def drone_student(tag):
    import train_rallynet as TR
    model = TR.load_model(tag)
    R = {tag: {}}
    for m, r in drone_rallies():
        refs = drone_refs(m, r)
        p = os.path.join(TR.OBS, "drone", f"{m}__{r}.npz")
        if refs is None or not os.path.exists(p):
            continue
        top, off, sh, pl = refs
        f = TR.load_feats(p)
        o = TR.infer(model, f)
        feet = o["joints"][:, :, [15, 16], :2].mean(2)
        feet = np.where((f["present"] > 0)[..., None], feet, np.nan)
        R[tag][f"{m}__{r}"] = drone_score(top, sh, pl, shuttle=o["shuttle"], feet=feet)
    dump_drone(R)


def gvhmr_feet(path, T=None):
    """GVHMR output (compact npz from tools/gvhmr_extract.py): per player frames + court-frame SMPL-24 ankles (GVHMR's
    own placement: its camera-frame joints mapped to the court with our calibration). -> (T, n, 3) mid-ankles."""
    v = np.load(path)
    n = int(v["n_players"])
    if n == 0:
        return None
    T = int(v["n_frames"]) if T is None else T
    A = np.full((T, n, 3), np.nan)
    for i in range(n):
        f = v[f"p{i}_frames"]
        ok = f < T
        A[f[ok], i] = v[f"p{i}_ankles"][ok].mean(1)
    return A


def gvhmr_cctv(nw=8):
    R = {}
    for uid in test_uids():
        p = os.path.join(ROOT, "outputs", "gvhmr", "cctv", uid + ".npz")
        refs = side_refs(uid)
        if refs is None or not os.path.exists(p):
            continue
        side, shift, sh, pl = refs
        A = gvhmr_feet(p)
        if A is not None:
            R[uid] = score(side, sh, pl, players=A)
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, "cctv_gvhmr.json"), "w"))
    s = [v["player"] for v in R.values() if v.get("player") is not None]
    print(f"gvhmr    player   {len(s)} rallies, median {np.median(s):.3f} m, coverage {np.mean([v['player_cov'] for v in R.values() if 'player_cov' in v]):.2f}", flush=True)


def drone_gvhmr():
    R = {"gvhmr": {}}
    for m, r in drone_rallies():
        refs = drone_refs(m, r)
        p = os.path.join(ROOT, "outputs", "gvhmr", "drone", f"{m}__{r}.npz")
        if refs is None or not os.path.exists(p):
            continue
        top, off, sh, pl = refs
        A = gvhmr_feet(p)
        if A is not None:
            R["gvhmr"][f"{m}__{r}"] = drone_score(top, sh, pl, feet=A[..., :2])
    dump_drone(R)


def dump_drone(R):
    os.makedirs(OUT, exist_ok=True)
    for name, d in R.items():
        json.dump(d, open(os.path.join(OUT, f"drone_{name}.json"), "w"))
        for k in ("shuttle", "player"):
            s = [v[k] for v in d.values() if v.get(k) is not None]
            if s:
                print(f"drone {name:8s} {k:8s} {len(s)} rallies, median {np.median(s):.3f} m, coverage {np.mean([v[k + '_cov'] for v in d.values() if k + '_cov' in v]):.2f}", flush=True)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "baselines"
    if stage == "student":
        student(os.environ.get("TAG", "t3"))
        return
    if stage == "drone_student":
        drone_student(os.environ.get("TAG", "t3"))
        return
    if stage == "freetrack":
        with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as ex:
            R = dict(ex.map(ft_one, test_uids()))
        print("errors", sum("error" in v for v in R.values()), [v for v in R.values() if "error" in v][:3], flush=True)
        dump(R, ("ft",))
        return
    if stage == "gvhmr":
        gvhmr_cctv()
        return
    if stage == "drone_gvhmr":
        drone_gvhmr()
        return
    if stage == "drone_tram":
        drone_tram()
        return
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    uids = test_uids()
    R = {}
    if os.environ.get("REDO_ERRORS") == "1":  # keep finished rallies, redo those missing from every method file
        old = [json.load(open(os.path.join(OUT, f"cctv_{m}.json"))) for m in ("mt", "v1", "v2fit", "tram", "v1p")]
        for u in uids:
            if all(u in o for o in old):
                R[u] = {m: o[u] for m, o in zip(("mt", "v1", "v2fit", "tram", "v1p"), old)}
        uids = [u for u in uids if u not in R]
        print("redo", len(uids), flush=True)
    with ProcessPoolExecutor(nw) as ex:
        R.update(dict(ex.map(baselines_one, uids)))
    print("errors", sum("error" in v for v in R.values()), [v for v in R.values() if "error" in v][:3], flush=True)
    dump(R, ("mt", "v1", "v2fit", "tram", "v1p"))


if __name__ == "__main__":
    main()
