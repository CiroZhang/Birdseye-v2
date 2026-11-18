"""Training examples from FineBadminton-20K rally clips (non-benchmark matches only):
  hits.jsonl   hit-detector candidates (birdseye_v2.hitclf features) labelled hit / not hit
  shots.jsonl  one row per labelled hit: shot features from the 3D fit with the labelled hit times
               (birdseye_v2.fbfeat) + labels: type, area (hitter's zone), hitter, next type, next area
usage: python tools/fb_build_train.py   -> outputs/fb20k/{hits,shots}.jsonl"""
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2 import shuttle3d
from birdseye_v2.broadcast import FPS, load_clip, native_frames, native_index
from birdseye_v2.fbfeat import STROKE_NAMES, shot_rows
from birdseye_v2.hitclf import candidate_features
from birdseye_v2.pipeline import reconstruct_rally
from birdseye_v2.shuttle3d import clean_track, landing_frame

CLIPS = os.path.join(ROOT, "data", "fb20k", "clips")
OUT = os.path.join(ROOT, "outputs", "fb20k")


def build(uid):
    try:
        lab = json.load(open(os.path.join(CLIPS, uid + ".json"), encoding="utf8"))
        cam, f, px, poses, on = load_clip(ROOT, uid, os.path.join(CLIPS, uid + ".mp4"), out="fb20k")
        if len(f) < 10:
            return None
        shuttle3d.FPS = FPS
        keep = native_frames(os.path.join(CLIPS, uid + ".mp4"), os.path.join(OUT, "native", uid + ".json"))
        hits = [native_index(keep, h["frame"]) for h in lab["hits"]]
        # hit-detector candidates on the cleaned track (as at test time)
        lf = landing_frame(f, px)
        fc, pc = clean_track(f, px, [lf - 1] if lf is not None else [f[-1]])
        cands = candidate_features(cam, fc, pc, poses)
        hrows = [dict(c, label=int(any(abs(c["tau"] - h) <= 3 for h in hits)), uid=uid) for c in cands]
        # shots with the labelled hit times
        inside = [i for i, h in enumerate(hits) if f[0] - 3 <= h <= f[-1]]
        if len(inside) < 1:
            return {"hits": hrows, "shots": []}
        rec = reconstruct_rally(cam, f, px, poses, hits=[float(hits[i]) for i in inside])
        sides = []
        for a, _ in rec["fit"].shots:  # label of the hit each fitted shot starts at
            j = inside[int(np.argmin([abs(a - hits[i]) for i in inside]))]
            sides.append(j)
        srows = []
        for r, j in zip(shot_rows(rec, FPS, sides=[lab["hits"][j]["hitter"] for j in sides]), sides):
            h = lab["hits"][j]
            nxt = lab["hits"][j + 1] if j + 1 < len(lab["hits"]) else None
            srows.append(dict(r, uid=uid, match=lab["match"], hit_index=j, n_hits=len(lab["hits"]),
                              type=STROKE_NAMES.get(h["type"], h["type"]), area=h["area"],
                              next_type=STROKE_NAMES.get(nxt["type"], nxt["type"]) if nxt else None,
                              next_area=nxt["area"] if nxt else None, point=h["point"]))
        return {"hits": hrows, "shots": srows}
    except Exception as e:
        print("FAILED", uid, repr(e), flush=True)
        return None


if __name__ == "__main__":
    uids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CLIPS, "*.mp4"))
                  if os.path.exists(os.path.join(OUT, "pose", os.path.basename(p)[:-4] + ".npz"))
                  and os.path.exists(os.path.join(OUT, "calib", os.path.basename(p)[:-4] + ".json")))
    nh = ns = 0
    with open(os.path.join(OUT, "hits.jsonl"), "w") as fh, open(os.path.join(OUT, "shots.jsonl"), "w", encoding="utf8") as fs, \
            ProcessPoolExecutor(14) as ex:
        for res in ex.map(build, uids, chunksize=2):
            if not res:
                continue
            for r in res["hits"]:
                fh.write(json.dumps(r) + "\n"); nh += 1
            for r in res["shots"]:
                fs.write(json.dumps(r, ensure_ascii=False) + "\n"); ns += 1
    print("clips", len(uids), "hit candidates", nh, "shots", ns)
