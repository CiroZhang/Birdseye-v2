"""Shuttle candidates in the player cameras without a learned detector: the motion-blurred streak.

In the player cameras' close-ups the shuttle is a large, bright, motion-blurred streak -- not the small dot TrackNet
was trained on, so TrackNet misses most of them (measured: ~15% found where the shuttle is in view). Here the camera
pan is cancelled with the calibrated rotation + zoom homographies (pcam/calib), and a pixel is a candidate where frame
k is brighter than BOTH neighbours k-GAP and k+GAP warped into its view (double difference: present only now -- a fast
object, not the moving camera or a slow player). Blobs are ranked by contrast x sqrt(area); up to MAXK per frame.
The teacher associates them with the TV ray itself (cctv_teach.guided_pass), so clutter here is cheap.
Output outputs/cctv_<view>/shuttle_streak/<uid>.json {frames, peaks [[x, y, score], ...]} (full-res px), as
pcam_shuttle.py.
usage: python tools/pcam_streak.py <view> [uid ...]   (env NPROC)"""
import json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FULL = (864, 486)
GAP = 2           # frames to each neighbour
THR = 22          # grey levels brighter than both neighbours
MAXK = 6
AREA = (6, 1500)  # px
MASK = os.environ.get("MASK", "1") == "1"  # drop blobs on the players' bodies (player-camera pose); the racket stays open
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def body_mask(kps):
    """Thick limbs, torso and head of every person (17 COCO keypoints, full-res px) -> uint8 mask."""
    m = np.zeros((FULL[1], FULL[0]), np.uint8)
    for k in kps:
        ok = k[:, 2] > 0.3
        if ok.sum() < 5:
            continue
        ys = k[ok, 1]
        w = int(max(8, 0.11 * (ys.max() - ys.min())))
        for a, b in LIMBS:
            if ok[a] and ok[b]:
                cv2.line(m, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)), 1, w)
        torso = [i for i in (5, 6, 12, 11) if ok[i]]
        if len(torso) >= 3:
            cv2.fillPoly(m, [k[torso, :2].astype(np.int32)], 1)
        if ok[0]:
            cv2.circle(m, tuple(k[0, :2].astype(int)), int(1.4 * w), 1, -1)
    return m


def homographies(z):
    R = Rotation.from_rotvec(z["rvec"]).as_matrix()
    K = np.zeros((len(R), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = z["f"]
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = FULL[0] / 2, FULL[1] / 2, 1
    return K @ R


def run(view, uid):
    z = np.load(os.path.join(ROOT, "outputs", "cctv_" + view, "pcam", "calib", uid + ".npz"))
    cap = cv2.VideoCapture(os.path.join(ROOT, "data", "cctv", "clips_" + view, uid + ".mp4"))
    G = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        G.append(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.int16))
    n = min(len(G), len(z["f"]))
    pose = None
    pp = os.path.join(ROOT, "outputs", "cctv_" + view, "pose", uid + ".npz")
    if MASK and os.path.exists(pp):
        pz = np.load(pp); ci = np.r_[0, np.cumsum(pz["counts"])]
        pose = [pz["kp"][ci[i]:ci[i + 1]] for i in range(len(pz["counts"]))]
    KR = homographies(z); inv = np.linalg.inv(KR)
    frames, peaks = [], []
    for k in range(GAP, n - GAP):
        cur = G[k]
        nb = []
        for a in (k - GAP, k + GAP):
            H = KR[k] @ inv[a]
            nb.append(cv2.warpPerspective(G[a].astype(np.uint8), H, FULL, flags=cv2.INTER_LINEAR, borderValue=255).astype(np.int16))
        d = np.minimum(cur - nb[0], cur - nb[1])
        m = (d > THR).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnt, lab, st, cen = cv2.connectedComponentsWithStats(m)
        bm = body_mask(pose[k]) if pose is not None and k < len(pose) else None
        cands = []
        for i in range(1, cnt):
            area = st[i, cv2.CC_STAT_AREA]
            if not AREA[0] <= area <= AREA[1]:
                continue
            if bm is not None and bm[int(cen[i][1]), int(cen[i][0])]:
                continue
            c = float(d[lab == i].mean())
            cands.append((float(cen[i][0]), float(cen[i][1]), c * np.sqrt(area)))
        cands.sort(key=lambda r: -r[2])
        if cands:
            frames.append(k); peaks.append([[round(a, 1), round(b, 1), round(s, 1)] for a, b, s in cands[:MAXK]])
    return {"frames": frames, "peaks": peaks}


def one(args):
    view, uid = args
    od = os.path.join(ROOT, "outputs", "cctv_" + view, "shuttle_streak" + ("_m" if MASK else ""))
    dst = os.path.join(od, uid + ".json")
    if os.path.exists(dst) or not os.path.exists(os.path.join(ROOT, "outputs", "cctv_" + view, "pcam", "calib", uid + ".npz")):
        return uid, "skip"
    try:
        r = run(view, uid)
    except Exception as e:
        return uid, "FAILED " + repr(e)[:200]
    os.makedirs(od, exist_ok=True)
    json.dump(r, open(dst, "w"))
    return uid, f"{len(r['frames'])} frames with candidates"


if __name__ == "__main__":
    view = sys.argv[1]
    uids = sys.argv[2:] or open(os.path.join(ROOT, "outputs", "cctv_pcam_dev_uids.txt")).read().split()
    with Pool(int(os.environ.get("NPROC", "6"))) as p:
        for r in p.imap_unordered(one, [(view, u) for u in uids]):
            print(*r, flush=True)
