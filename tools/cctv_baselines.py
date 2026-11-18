"""TV-only shuttle baselines on the held-out CCTV match, scored like the student (per-rally median 3D distance to
the two-view TV + side triangulation, outputs/cctv/tri). Same inputs for every method, all from the TV view: its
BirdsCourt camera, the refined TrackNet track, YOLOv8-pose people (pose1280), and hits detected from the TV track
(birdseye_v2.shuttle3d.detect_hits, then the same shot splitting as tools/eval_shuttle.py's detected mode).
  v1   Birdseye v1 per-shot fit with hand anchors (third_party/birdseye_v1)
  mt   MonoTrack's per-shot fit (birdseye_v2/monotrack_fit.py)
  v2   per-rally sparse-hit physics fit (birdseye_v2.shuttle3d.fit_rally)

out  outputs/peval_k/base_<method>_cctv.json  {uid: {shuttle: median error (m), coverage}}
usage: python tools/cctv_baselines.py [n_workers]"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
FPS = 25.0
METHODS = ("v1", "mt", "v2")


def one(uid):
    from birdseye_v2 import shuttle3d as S3
    S3.FPS = FPS
    import eval_shuttle as ES
    ES.FPS = FPS
    from birdseye_v2.court import Camera
    from birdseye_v2.players import load_pose, foot_point
    from birdseye_v2.monotrack_fit import reconstruct as mt_reconstruct
    import cctv_tri as CT
    try:
        cam = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "cctv", "calib_bc", uid + ".json")))["cam"])
        tr = CT.track(os.path.join(ROOT, "outputs", "cctv", "refined", uid + ".json"))
        if len(tr) < 10:
            return uid, {"error": "short track"}
        frames = np.array(sorted(tr), float)
        px = np.array([tr[k] for k in sorted(tr)], float)
        poses = load_pose(os.path.join(ROOT, "outputs", "cctv", "pose1280", uid + ".npz"))
        gt = np.array(json.load(open(os.path.join(ROOT, "outputs", "cctv", "tri", uid + ".json")))["shuttle"], float).reshape(-1, 4)
        gt_t, gt_X = gt[:, 0], gt[:, 1:]
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

        out = {}
        preds = {}
        preds["v1"] = ES.fit_v1(cam, frames, px, shots, hands)(gt_t)
        preds["mt"] = mt_reconstruct(cam, frames, px, [int(round(a)) for a, _ in shots] + [int(round(shots[-1][1]))], foot_at, FPS)(gt_t)
        try:
            preds["v2"] = S3.fit_rally(cam, frames, px, shots, cands, use_net=False).position(gt_t, rest_until=frames[-1])
        except Exception as e:
            preds["v2"] = np.full_like(gt_X, np.nan)
        for k, P in preds.items():
            e = np.linalg.norm(P - gt_X, axis=1)
            ok = np.isfinite(e)
            out[k] = {"shuttle": float(np.median(e[ok])) if ok.any() else None, "coverage": float(ok.mean()), "n": int(len(e))}
        return uid, out
    except Exception as ex:
        return uid, {"error": repr(ex)[:200]}


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    test = set(json.load(open(os.path.join(ROOT, "outputs", "cctv", "split.json")))["test"])
    uids = sorted(u for u in (p[:-5] for p in os.listdir(os.path.join(ROOT, "outputs", "cctv", "tri")) if p.endswith(".json")) if u in test)
    with ProcessPoolExecutor(nw) as ex:
        R = dict(ex.map(one, uids))
    for m in METHODS:
        d = {u: v[m] for u, v in R.items() if m in v}
        json.dump(d, open(os.path.join(ROOT, "outputs", "peval_k", f"base_{m}_cctv.json"), "w"))
        s = np.array([v["shuttle"] for v in d.values() if v["shuttle"] is not None])
        print(f"{m}: {len(d)} rallies, median of rally medians {np.median(s):.3f} m, coverage {np.mean([v['coverage'] for v in d.values()]):.2f}", flush=True)
    print("errors", sum("error" in v for v in R.values()), [v for v in R.values() if "error" in v][:3])


if __name__ == "__main__":
    main()
