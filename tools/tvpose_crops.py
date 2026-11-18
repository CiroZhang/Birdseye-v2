"""TV image crops around each player's upper body, for the image branch of the TV pose student (tools/tvpose_train.py).

Per track of a tools/tvpose_data.py pack (same order), every STRIDE-th frame of the TV clip: a square crop centred on
the VIMO chest (spine3), 2.2 m wide at the player's depth (the racket reaches ~0.7 m past the hand), resized to
SIZE px, JPEG-encoded. Output OUT.pkl: list per track of {frame index within the track: jpeg bytes}.

usage: python tools/tvpose_crops.py PACK.npz OUT.pkl [--stride 2] [--size 112] [--part i/n]
"""
import argparse, json, os, pickle, sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import body_mv as BM  # noqa: E402
from tvpose_data import tv_cam  # noqa: E402

WIDTH_M = 2.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack"); ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    pi, pn = map(int, a.part.split("/"))
    z = np.load(a.pack)
    off, uids, J_all, fr_all = z["off"], z["uid"], z["J"], z["frames"]
    by_uid = {}
    for i, u in enumerate(uids):
        by_uid.setdefault(str(u), []).append(i)
    todo = sorted(by_uid)[pi::pn]
    out = {}
    for n, uid in enumerate(todo):
        cam = tv_cam(uid)
        tr = by_uid[uid]
        want = {}  # clip frame -> [(track, row, centre px, half size px)]
        for i in tr:
            J, fr = J_all[off[i]:off[i + 1]], fr_all[off[i]:off[i + 1]]
            for r in range(0, len(fr), a.stride):
                if not np.isfinite(J[r]).all():
                    continue
                uv, dep = BM.project(cam, J[r, 9][None])
                if dep[0] < 0.5:
                    continue
                h = 0.5 * WIDTH_M * cam["f"] / dep[0]
                want.setdefault(int(fr[r]), []).append((i, r, uv[0], h))
        cap = cv2.VideoCapture(os.path.join(ROOT, "data", "cctv", "clips", uid + ".mp4"))
        k, last = 0, max(want) if want else -1
        res = {i: {} for i in tr}
        while k <= last:
            ok, im = cap.read()
            if not ok:
                break
            for i, r, c, h in want.get(k, []):
                x0, y0, s = int(round(c[0] - h)), int(round(c[1] - h)), int(round(2 * h))
                if s < 8:
                    continue
                pad = cv2.copyMakeBorder(im, s, s, s, s, cv2.BORDER_CONSTANT, value=0)
                crop = pad[y0 + s:y0 + 2 * s, x0 + s:x0 + 2 * s]
                crop = cv2.resize(crop, (a.size, a.size), interpolation=cv2.INTER_AREA)
                res[i][r] = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            k += 1
        out.update(res)
        if n % 50 == 0:
            print(n, len(todo), uid, sum(len(v) for v in res.values()), flush=True)
    pickle.dump(out, open(a.out, "wb"))
    print("saved", a.out, len(out), "tracks", sum(len(v) for v in out.values()), "crops")


if __name__ == "__main__":
    main()
