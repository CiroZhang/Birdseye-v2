"""Export rallies for the full-pipeline viewer (outputs/viewer/): everything the v2 pipeline sees and
produces from the BACK VIEW ONLY -- players (YOLOv8x-pose, lifted to 3D on the vertical plane
through each player's feet), rackets (YOLOv8x COCO class 38, lifted at the owner's depth), the
shuttle (TrackNetV3 detections + v2 3D fit with detected hits, learned hit detector) -- plus the
two-view ground truth for comparison. Writes a downscaled back-view clip per rally and data.json.
usage: python tools/export_viewer.py BD15/Rally-39 BD17/Rally-1 ..."""
import csv, json, os, sys

import cv2
import imageio_ffmpeg
import joblib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import NET_Y, Camera
from birdseye_v2.footlock import stabilize
from birdseye_v2.pipeline import rally_summary, reconstruct_rally
from birdseye_v2.players import lift_at_depth, load_pose, load_rackets, players_3d
sys.path.insert(0, os.path.join(ROOT, "tools"))
from eval_players import gt_players, match_err
from birdseye_v2.tracks import RACKET_HEAD, RACKET_LEN, _fill, rackets_for_tracks, smooth_track, track_players

OUT = os.path.join(ROOT, "outputs", os.environ.get("VIEWER_OUT", "viewer"))
CALIB = os.environ.get("DRONE_CALIB", "calib")  # calib_bc: BirdsCourt back camera (tools/drone_birdscourt.py)
BODY = os.environ.get("BODY", "yolo")  # "vimo": TRAM/VIMO SMPL bodies (tools/drone_vimo.py), shuttle anchored at their hands
DRONE_VIMO = os.environ.get("DRONE_VIMO", "drone_vimo")
VW, VH = 960, 540
PRE, POST = 15, 30  # frames of context around the shuttle track


def r3(p):
    return None if p is None or not np.isfinite(p).all() else [round(float(v), 3) for v in p]


def norm_px(uv, size):
    return [round(float(uv[0]) / size[0], 4), round(float(uv[1]) / size[1], 4)]


def face_normals(w, u, min_speed=1.0, sigma=2.0):
    """Racket-face normal per frame: a swung racket moves face-first, so the normal is the head's velocity
    with its along-shaft part removed (head speed >= min_speed m/s); while the racket is (nearly) still it
    keeps the last swing's face; before any swing NaN (the viewer then faces the camera)."""
    from scipy.ndimage import gaussian_filter1d
    head = w + 0.48 * u
    v = np.gradient(np.nan_to_num(head), axis=0) * 30.0
    v = gaussian_filter1d(v, sigma, axis=0)
    vp = v - np.sum(v * u, 1, keepdims=True) * u
    sp = np.linalg.norm(vp, axis=1)
    n = np.full_like(u, np.nan)
    last = None
    for i in range(len(u)):
        if np.isfinite(u[i]).all() and sp[i] >= min_speed:
            last = vp[i] / sp[i]
        if last is not None and np.isfinite(u[i]).all():
            m = last - (last @ u[i]) * u[i]  # keep it perpendicular to the current shaft
            if np.linalg.norm(m) > 1e-3:
                n[i] = m / np.linalg.norm(m)
    return n


def write_clip(src, dst, f0, f1):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    w = imageio_ffmpeg.write_frames(dst, (VW, VH), fps=30, codec="libx264", quality=None,
                                    output_params=["-crf", "30", "-preset", "slow", "-movflags", "+faststart"],
                                    macro_block_size=1)
    w.send(None)
    for _ in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        w.send(np.ascontiguousarray(cv2.cvtColor(cv2.resize(img, (VW, VH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
    w.close()
    cap.release()


def export(mr, hit_model):
    m, r = mr.split("/")
    g = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))
    off = g["sync"]["offset"]
    cam = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", CALIB, m, r + ".json")))["back"])
    VJ = np.load(os.path.join(ROOT, "outputs", DRONE_VIMO, m, r + ".npz"))["joints"] if BODY == "vimo" else None
    if VJ is not None:  # smooth ground path + planted feet pinned (birdseye_v2.footlock)
        VJ = np.stack([stabilize(J) for J in VJ])
    if VJ is not None:  # hand anchors from the VIMO bodies, as in tools/eval_shuttle.py ANCHOR=vimo
        import eval_shuttle as ES
        from birdseye_v2 import pipeline as PL
        ES.VIMO_J = VJ
        PL._candidates = lambda c, po, fr, px, t, rackets=None: ES.hand_cands_for(c, po, fr, px, t, max_px=1e9, rackets=rackets)
    poses = load_pose(os.path.join(ROOT, "outputs", "pose", m, r + ".npz"))
    rk_path = os.path.join(ROOT, "outputs", "racket", m, r + ".npz")
    rackets = load_rackets(rk_path) if os.path.exists(rk_path) else None
    bt_f = np.array(g["back_track"]["frames"], float)
    bt_px = np.array(g["back_track"]["px"], float).reshape(-1, 2)

    # the automatic pipeline: back view only, detected hits
    rec = reconstruct_rally(cam, bt_f, bt_px, poses, hit_model=hit_model)
    fit = rec["fit"]
    labels = [int(x["frame"]) + off for x in csv.DictReader(open(os.path.join(ROOT, "data", "datacsv", m, r, "1.csv")))
              if x["status"] not in ("0", "")]

    n = len(poses)
    f0, f1 = max(0, int(bt_f[0]) - PRE), min(n - 1, int(bt_f[-1]) + POST)
    frames = np.arange(f0, f1 + 1)
    v2 = fit.position(frames.astype(float), rest_until=f1)
    v2_px = cam.project(np.nan_to_num(v2))

    gt_t = np.array([p["frame"] for p in g["points"]], float) + off
    gt_X = np.array([p["X"] for p in g["points"]]).reshape(-1, 3)
    e = np.linalg.norm(fit.position(gt_t, rest_until=f1) - gt_X, axis=1)

    det = {int(round(f)): uv for f, uv in zip(bt_f, bt_px)}
    gt_by = {}
    for t, X in zip(gt_t, gt_X):
        k = int(round(t))
        if abs(t - k) <= 0.5:
            gt_by[k] = X
    # top-view annotated players (top clock) -> back clock, for comparison only
    top = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", CALIB, m, r + ".json")))["top"])
    gp = {int(round(ft + off)): P for ft, P in gt_players(m, r, top).items()}

    # players: tracked identities, gap-filled + smoothed keypoints, lifted at a smoothed foot depth;
    # rackets: fixed-size model held in the racket hand, direction from detections (else forearm)
    tracks = [smooth_track(cam, tr) for tr in track_players(cam, poses, frames)]
    boxes = [rackets[f] if rackets is not None and f < len(rackets) else np.zeros((0, 5)) for f in frames]
    rackets_for_tracks(cam, tracks, boxes)
    if VJ is not None:  # replace the lifted YOLO skeletons with the SMPL-24 VIMO joints
        from fb_full_example import rackets as vimo_rackets
        tracks = []
        for J in VJ:
            J = J[np.clip(frames, 0, J.shape[0] - 1)].copy()
            J[frames >= VJ.shape[1]] = np.nan
            ok = np.isfinite(J[:, [7, 8]]).all((1, 2))
            if ok.sum() < 10:
                continue
            uv = np.full(J.shape[:2] + (2,), np.nan)
            uv[ok] = cam.project(J[ok].reshape(-1, 3)).reshape(-1, 24, 2)
            w, u, near, _ = vimo_rackets(cam, J, boxes)
            u[~ok] = np.nan
            tracks.append({"X": J, "uv": uv, "foot": np.where(ok[:, None], J[:, [7, 8], :2].mean(1), np.nan),
                           "racket_w": w, "racket_u": u, "racket_near": near, "racket_n": face_normals(w, u)})

    # shuttle: before the first shot the shuttle is in the server's hand -> hold the serve point
    first = np.where(np.isfinite(v2).all(1))[0]
    if len(first):
        v2[:first[0]] = v2[first[0]]
        v2_px = cam.project(np.nan_to_num(v2))
    # ground truth: fill short gaps (both views missed it for a few frames)
    gt_arr = np.full((len(frames), 3), np.nan)
    for k, X in gt_by.items():
        if f0 <= k <= f1:
            gt_arr[k - f0] = X
    gt_arr = _fill(np.arange(len(frames)), gt_arr, max_gap=6)

    per = []
    for i, f in enumerate(frames):
        P, Rk = [], []
        for tid, tr in enumerate(tracks):
            if not np.isfinite(tr["foot"][i]).all() or not np.isfinite(tr["X"][i]).any():
                continue
            P.append({"id": tid, "s": "n" if tr["foot"][i][1] < NET_Y else "f",
                      "j": [r3(x) if np.isfinite(x).all() else None for x in tr["X"][i]],
                      "k": [norm_px(x, cam.size) if np.isfinite(x).all() else None for x in tr["uv"][i]]})
            w, u = tr["racket_w"][i], tr["racket_u"][i]
            if np.isfinite(w).all() and np.isfinite(u).all():
                butt, tip = w - 0.08 * u, w + (RACKET_LEN - 0.08) * u
                nv = tr["racket_n"][i] if "racket_n" in tr else None
                Rk.append({"id": tid, "w": r3(w), "u": r3(u), "seen": bool(tr["racket_near"][i]),
                           **({"n": r3(nv)} if nv is not None and np.isfinite(nv).all() else {}),
                           "px": [norm_px(x, cam.size) for x in cam.project(np.array([butt, tip]))]})
        per.append({
            "p": P, "r": Rk,
            "rb": [norm_px(b[:2], cam.size) + norm_px(b[2:4], cam.size) for b in boxes[i] if b[4] >= 0.25],
            "x": r3(v2[i]),
            "xp": norm_px(v2_px[i], cam.size) if np.isfinite(v2[i]).all() else None,
            "d": norm_px(det[f], cam.size) if f in det else None,
            "g": r3(gt_arr[i]) if np.isfinite(gt_arr[i]).all() else None,
            "pg": [[round(float(a), 2), round(float(b), 2)] for a, b in gp.get(int(f), [])],
        })

    pe = []
    for i, f in enumerate(frames):
        P = gp.get(int(f))
        if P is not None and len(P):
            feet = np.array([t["foot"][i] for t in tracks if np.isfinite(t["foot"][i]).all()]).reshape(-1, 2)
            pe += [x for x in match_err(P, feet) if x is not None]
    player_err = round(float(np.median(pe)), 3) if pe else None

    summ = rally_summary(rec)
    hits = rec["hits"]
    tp = sum(any(abs(h - l) <= 3 for h in hits) for l in labels)
    ck = lambda t: round(float(t) - f0, 2)  # clip-relative frame
    shots = []
    for s, (a, b) in zip(summ, fit.shots):
        ts = np.arange(a, b + 1e-9, 0.5)
        shots.append({"t0": ck(a), "t1": ck(b), "side": s["hitter_side"], "speed": s["speed_kmh"],
                      "apex": s["apex_m"], "anchored": s["anchored"],
                      "path": [r3(x) for x in fit.position(ts)]})
    os.makedirs(OUT, exist_ok=True)
    clip = f"{m}_{r}.mp4"
    write_clip(os.path.join(ROOT, "data", "videos", m, r, "(1).mp4"), os.path.join(OUT, clip), f0, f1)

    R = cam.R
    meta = {
        "id": f"{m} {r}", "clip": clip, "skel": "smpl24" if VJ is not None else "coco17", "n": len(frames), "f0": int(f0),
        "cam": {"c": r3(cam.center), "fwd": r3(R.T @ [0, 0, 1.0]), "down": r3(R.T @ [0, 1.0, 0]),
                "vfov": round(float(np.degrees(2 * np.arctan(cam.size[1] / 2 / cam.f))), 2),
                "aspect": round(cam.size[0] / cam.size[1], 4)},
        "shots": shots, "labels": [ck(l) for l in labels],
        "gt_path": [[ck(t)] + r3(X) for t, X in zip(gt_t, gt_X)],
        "racket": {"len": RACKET_LEN, "head": list(RACKET_HEAD), "grip": 0.08},
        "metrics": {"player_err_median": player_err,
                    "err_median": round(float(np.nanmedian(e)), 3), "err_mean": round(float(np.nanmean(e)), 3),
                    "n_gt": int(len(e)), "hits_detected": len(hits), "hits_labeled": len(labels),
                    "hit_recall": round(tp / max(len(labels), 1), 2),
                    "has_racket": rackets is not None},
        "frames": per,
    }
    print(mr, meta["metrics"], flush=True)
    return meta


if __name__ == "__main__":
    hm = joblib.load(os.path.join(ROOT, "outputs", "hitclf.joblib"))
    data = [export(mr, hm) for mr in sys.argv[1:]]
    json.dump(data, open(os.path.join(OUT, "data.json"), "w"), separators=(",", ":"))
    print("wrote", os.path.join(OUT, "data.json"), os.path.getsize(os.path.join(OUT, "data.json")) // 1024, "KB")
