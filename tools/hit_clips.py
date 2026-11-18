"""Training data for the video hit model: per rally one synced 2x2 mosaic video on the TV clock (side | player cam 1 /
player cam 2 | TV), 256x144 per view, 25 fps, keyframe every 4 frames (fast random access), plus labels:
  pos   -- physics-verified teacher hits (confident flight before and after), TV frame (float) and side (0 near, 1 far)
  neg   -- TV frames inside confident flight, > 8 frames from every teacher hit (no hit possible there)
Output outputs/cctv/hitclips/<uid>.mp4 and labels.json. Rallies in EXCLUDE (the hand answer key) are skipped.
usage: python tools/hit_clips.py [nproc]"""
import glob, json, os, sys
from multiprocessing import Pool

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.environ.get("HC_OUT", os.path.join(ROOT, "outputs", "cctv", "hitclips"))
TEACH = os.path.join(ROOT, "outputs", "cctv", "teach")
W, H = 256, 144
PCAM_SYNC = {"p1": -7.5, "p2": -6.9}
EXCLUDE = os.environ.get("HC_EXCLUDE", "")
LEAD = int(os.environ.get("HC_LEAD", "0"))   # extra frames before the clip start for side/player cams (the sync shift
                                             # leaves their first frames empty otherwise), cut from the session stream
ONLY = os.environ.get("HC_ONLY", "")         # optional json list of uids to build


def frames(d, uid):
    cap = cv2.VideoCapture(f"{ROOT}/data/cctv/{d}/{uid}.mp4"); F = []
    while True:
        ok, im = cap.read()
        if not ok:
            return F
        F.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))


def frames_lead(view, uid, n):
    """LEAD frames before the rally clip plus the clip itself (n frames), same timing as the cut in cctv_rallies.py"""
    sess = uid.rsplit("_", 1)[0]; m = json.load(open(f"{ROOT}/data/cctv/{sess}/rallies/{uid}/meta.json"))
    st = m["tv_start_s"] + m["offsets_s"][view]
    r = imageio_ffmpeg.read_frames(f"{ROOT}/data/cctv/{sess}/{m['views'][view]}.mp4", input_params=["-ss", f"{st - LEAD / 25:.3f}"],
                                   output_params=["-t", f"{(LEAD + n) / 25:.3f}", "-r", "25"])
    w, h = next(r)["size"]
    return [cv2.resize(np.frombuffer(f, np.uint8).reshape(h, w, 3)[:, :, ::-1], (W, H), interpolation=cv2.INTER_AREA) for f in r]


def labels(uid):
    o = json.load(open(os.path.join(TEACH, uid + ".json")))
    fr = np.array(o["frames"], float); sd = np.array(o["sd"], float); P = np.array(o["P"], float); hits = np.array(o["hits"], float)
    conf = {int(f): s < 0.06 for f, s in zip(fr, sd)}
    ok = lambda fs: np.mean([conf.get(int(f), False) for f in fs]) >= 0.6
    pos = []
    for h in hits:
        if ok(np.arange(np.floor(h) - 8, np.floor(h) - 1)) and ok(np.arange(np.ceil(h) + 2, np.ceil(h) + 9)):
            y = np.interp(h, fr, P[:, 1]); pos.append([float(h), int(y > 6.7)])
    neg = [int(f) for f, c in conf.items() if c and (not len(hits) or np.abs(hits - f).min() > 8)]
    return pos, neg


def one(uid):
    dst = os.path.join(OUT, uid + ".mp4")
    try:
        pos, neg = labels(uid)
        if not os.path.exists(dst):
            sh = json.load(open(f"{ROOT}/outputs/cctv/tri/{uid}.json"))["shift"]
            TV = frames("clips", uid)
            if LEAD:
                SD, P1, P2 = (frames_lead(v, uid, len(TV)) for v in ("side", "p1", "p2"))
            else:
                SD, P1, P2 = frames("clips_side", uid), frames("clips_p1", uid), frames("clips_p2", uid)
            blank = np.zeros((H, W, 3), np.uint8)
            get = lambda F, k: F[k + LEAD] if 0 <= k + LEAD < len(F) else blank
            w = imageio_ffmpeg.write_frames(dst, (2 * W, 2 * H), fps=25, codec="libx264", quality=None,
                                            output_params=["-crf", "23", "-preset", "fast", "-g", "4", "-bf", "0", "-pix_fmt", "yuv420p"])
            w.send(None)
            for k in range(len(TV)):
                m = np.vstack([np.hstack([get(SD, k + sh), get(P1, int(round(k + PCAM_SYNC["p1"])))]),
                               np.hstack([get(P2, int(round(k + PCAM_SYNC["p2"]))), TV[k]])])
                w.send(np.ascontiguousarray(m[:, :, ::-1]))
            w.close()
            n = len(TV)
        else:
            n = int(cv2.VideoCapture(dst).get(cv2.CAP_PROP_FRAME_COUNT))
        return uid, {"n": n, "pos": pos, "neg": neg}
    except Exception as e:
        return uid, {"error": str(e)}


def main():
    os.makedirs(OUT, exist_ok=True)
    ex = set(json.load(open(EXCLUDE))) if EXCLUDE else set()
    uids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(TEACH, "*.json")))
    uids = [u for u in uids if u not in ex]
    if ONLY:
        keep = set(json.load(open(ONLY))); uids = [u for u in uids if u in keep]
    lab = {}
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        for i, (u, d) in enumerate(pool.imap_unordered(one, uids)):
            lab[u] = d
            if i % 100 == 0 or "error" in d:
                print(i, u, d.get("error", f"pos {len(d.get('pos', []))} neg {len(d.get('neg', []))}"), flush=True)
    json.dump(lab, open(os.path.join(OUT, "labels.json"), "w"))
    print("DONE", len(lab), "rallies,", sum(len(d.get("pos", [])) for d in lab.values()), "positives", flush=True)


if __name__ == "__main__":
    main()
