"""Calibrate a session's fixed side camera once: the median of side-view frames sampled across the session's rallies
(players and officials move, the court does not), fitted with the view-agnostic court-line fit (birdseye_v2.courtfit).

Output: outputs/cctv/side_cam_<session>.json {cam, stats} + side_fit_<session>.jpg (overlay to check by eye)
usage: python tools/cctv_sidecal.py <session>   (rallies under data/cctv/<session>/rallies, cut by cctv_rallies.py)"""
import glob, json, os, sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2 import courtfit as CF  # noqa: E402
from birdseye_v2.court import draw_court  # noqa: E402


def main():
    sess = sys.argv[1]
    clips = sorted(glob.glob(os.path.join(ROOT, "data", "cctv", sess, "rallies", "*", "side.mp4")))
    frames = []
    for p in clips[:: max(1, len(clips) // 60)]:
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, im = cap.read()
        if ok:
            frames.append(im)
    med = np.median(np.stack(frames), 0).astype(np.uint8)
    cam, st = CF.calibrate(med)
    out = os.path.join(ROOT, "outputs", "cctv")
    cv2.imwrite(os.path.join(out, f"side_median_{sess}.jpg"), med)
    if cam is None:
        print(sess, "side camera NOT calibrated:", st)
        sys.exit(1)
    json.dump({"cam": cam.to_dict(), "stats": {k: v for k, v in st.items() if isinstance(v, (int, float, str, bool))},
               "frames": len(frames)}, open(os.path.join(out, f"side_cam_{sess}.json"), "w"), indent=1)
    cv2.imwrite(os.path.join(out, f"side_fit_{sess}.jpg"), draw_court(med, cam, thick=1))
    print(sess, "side camera:", {k: st.get(k) for k in ("view", "score", "precision", "f")}, "from", len(frames), "frames")


if __name__ == "__main__":
    main()
