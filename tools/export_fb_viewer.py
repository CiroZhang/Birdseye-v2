"""Broadcast viewer data: FineBadminton-20K rallies the fine-tuned v1 never trained on, with VIMO bodies,
the fine-tuned hit/stroke predictions and the FineBadminton labels.

Per rally: 720p H.264 clip of the native (de-duplicated, 25 fps) frames, so video frame i = VIMO frame i;
SMPL-24 world joints per side; the same joints projected through the camera VIMO ran with (the clip's
calibration when trusted, else the nominal broadcast camera) for the video overlay; per-frame hit
probability per side; predicted and labeled hits.
usage: .venv_vimo/Scripts/python tools/export_fb_viewer.py [uid ...]  -> outputs/fb_viewer/"""
import json, os, sys

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
import v1_finetune as V  # noqa: E402
from birdseye_v2 import v1action as A  # noqa: E402
from birdseye_v2.broadcast import FPS  # noqa: E402
from birdseye_v2.court import Camera, init_from_corners  # noqa: E402

DST = os.path.join(ROOT, "outputs", "fb_viewer")
VW, VH = 960, 540
GROUPS = {"0011": "test", "0016": "test", "0055": "val", "0065": "val"}  # model split (outputs/v1ft)
TOL = 8  # 25 fps frames, as tools/hit_eval_fb.py


def nominal_camera():
    return init_from_corners([(219, 645), (1062, 648), (866, 278), (414, 278)], (1280, 720), 2000.0)


def write_clip(src, dst, keep):
    cap, want, i = cv2.VideoCapture(src), set(int(k) for k in keep), 0
    w = imageio_ffmpeg.write_frames(dst, (VW, VH), fps=FPS, codec="libx264", quality=None,
                                    output_params=["-crf", "28", "-preset", "slow", "-movflags", "+faststart"],
                                    macro_block_size=1)
    w.send(None)
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            w.send(np.ascontiguousarray(cv2.cvtColor(cv2.resize(im, (VW, VH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
        i += 1
    w.close()


def rnd(a, k=3):
    return None if not np.isfinite(a).all() else [round(float(x), k) for x in a]


def peaks(hp, thr):
    out = []
    for i in np.argsort(-hp):
        if hp[i] < thr:
            break
        if all(abs(i - p) >= 10 for p in out):
            out.append(int(i))
    return sorted(out)


def export(model, thr, uid):
    r = V.load_rally(uid)
    if r is None:
        return None
    v = np.load(os.path.join(V.VIMO, uid + ".npz"))
    keep, T = v["frames"], len(v["top"])
    cal = json.load(open(os.path.join(ROOT, "outputs", "fb20k", os.environ.get("CALIB_DIR", "calib_bc"), uid + ".json")))
    trusted = bool(v["trusted_calib"])
    cam = Camera.from_dict(cal["cam"]) if trusted else nominal_camera()
    pr = V.predict(model, r)
    to25 = lambda f30: f30 * FPS / 30.0  # noqa: E731
    pred, prob = [], {}
    for s in ("top", "bottom"):
        hp, sp = pr[s]
        idx25 = np.clip(np.round(np.arange(T) * 30.0 / FPS).astype(int), 0, len(hp) - 1)
        prob[s] = [round(float(x), 2) for x in hp[idx25]]
        for p in peaks(hp, thr):
            c = sp[p]
            pred.append({"f": round(to25(p), 1), "side": s, "type": V.CLASSES[int(c.argmax())], "conf": round(float(c.max()), 2),
                         "hit": round(float(hp[p]), 2)})
    pred.sort(key=lambda h: h["f"])
    gt = [{"f": round(to25(f), 1), "side": s, "type": t} for f, s, t in r["hits"]]
    # match predictions to labels (greedy by time gap, as the evaluation)
    pairs = sorted((abs(p["f"] - g["f"]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt) if abs(p["f"] - g["f"]) <= TOL)
    up, ug = set(), set()
    for _, i, j in pairs:
        if i not in up and j not in ug:
            up.add(i); ug.add(j); pred[i]["match"] = j; gt[j]["match"] = i
    tp = len(up)
    frames = []
    for i in range(T):
        F = {}
        for s, key in (("top", "t"), ("bottom", "b")):
            J = v[s][i]
            if np.isfinite(J).all():
                uv = cam.project(J) / np.array(cam.size, float)
                F[key] = {"j": [rnd(x) for x in J], "k": [[round(float(a), 4), round(float(b), 4)] for a, b in uv]}
        frames.append(F)
    os.makedirs(DST, exist_ok=True)
    write_clip(os.path.join(V.CLIPS, uid + ".mp4"), os.path.join(DST, uid + ".mp4"), keep[:T])
    c = cam.center
    fwd, down = cam.R.T @ np.array([0, 0, 1.0]), cam.R.T @ np.array([0, 1.0, 0])
    side_ok = sum(pred[i]["side"] == gt[j]["side"] for i, j in ((i, pred[i]["match"]) for i in up))
    type_ok = sum(pred[i]["type"] == gt[j]["type"] for i, j in ((i, pred[i]["match"]) for i in up))
    return {"id": uid, "group": GROUPS.get(uid[:4], ""), "trusted": trusted, "clip": uid + ".mp4", "n": T, "fps": FPS,
            "cam": {"c": rnd(c), "fwd": rnd(fwd, 4), "down": rnd(down, 4),
                    "vfov": round(float(np.degrees(2 * np.arctan(cam.size[1] / 2 / cam.f))), 2)},
            "pred": pred, "gt": gt, "prob": prob, "frames": frames,
            "metrics": {"tp": tp, "n_pred": len(pred), "n_gt": len(gt), "side_ok": side_ok, "type_ok": type_ok}}


def pick(n_per=2):
    out = []
    for m in GROUPS:
        cands = []
        for f in sorted(os.listdir(V.VIMO)):
            if not f.startswith(m + "_"):
                continue
            uid = f[:-4]
            lab = json.load(open(os.path.join(V.CLIPS, uid + ".json"), encoding="utf8"))
            n = len(np.load(os.path.join(V.VIMO, f))["top"])
            if 8 <= len(lab["hits"]) and n <= 450:
                cands.append(uid)
        out += cands[:: max(1, len(cands) // n_per)][:n_per]
    return out


if __name__ == "__main__":
    V.DEV = os.environ.get("DEV", V.DEV)
    ck = torch.load(os.path.join(ROOT, "outputs", "v1ft", "model.pt"), map_location="cpu")
    model = A.load(V.DEV)  # tiny model; DEV=cpu when the GPU is busy
    model.shot_head[-1] = nn.Linear(model.shot_head[-1].in_features, len(ck["classes"])).to(V.DEV)
    model.load_state_dict(ck["state"]); model.eval()
    uids = sys.argv[1:] or pick()
    data = [d for d in (export(model, ck["thr"], u) for u in uids) if d]
    for d in data:
        m = d["metrics"]
        print(d["id"], d["group"], f"{m['tp']}/{m['n_gt']} labeled hits found, {m['n_pred']} predicted, "
              f"side {m['side_ok']}/{m['tp']} type {m['type_ok']}/{m['tp']}", flush=True)
    json.dump(data, open(os.path.join(DST, "data.json"), "w"), separators=(",", ":"))
