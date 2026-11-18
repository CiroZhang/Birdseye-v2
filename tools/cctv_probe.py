"""First look at CCTV / Yangshipin multi-angle match videos (data/cctv/<match>/<view>.mp4): resolution, fps, length,
a contact sheet of frames spread over the video (to name the view), and whether the camera stays put -- the
fraction of a fixed edge map (the court's painted lines) that each sampled frame still overlaps.

Output: outputs/cctv/probe/<match>__<view>.jpg (contact sheet) + .json {fps, size, minutes, stable: [...]}
usage: python tools/cctv_probe.py [path.mp4 ...]   (no args = every mp4 under data/cctv)"""
import glob, json, os, sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "outputs", "cctv", "probe")
N_SHEET = 12  # frames on the contact sheet
N_STABLE = 60  # frames for the stability check


def edges(im):
    g = cv2.cvtColor(cv2.resize(im, (640, 360)), cv2.COLOR_BGR2GRAY)
    return cv2.dilate(cv2.Canny(g, 60, 160), np.ones((3, 3), np.uint8)) > 0


def probe(path):
    cap = cv2.VideoCapture(path)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if n <= 0:
        return {"error": "unreadable (still downloading?)"}

    def grab(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        return im if ok else None

    tiles = []
    for i in np.linspace(0, n - 1, N_SHEET + 2)[1:-1]:
        im = grab(i)
        if im is None:
            continue
        t = cv2.resize(im, (480, int(480 * size[1] / size[0])))
        cv2.putText(t, f"{i / fps / 60:.0f} min", (10, 30), 0, 0.9, (255, 255, 255), 2)
        tiles.append(t)
    sheet = None
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    # stable: lines that are present in most frames are the court; each frame's share of them
    E = [e for e in (grab(i) for i in np.linspace(0, n - 1, N_STABLE)) if e is not None]
    E = np.array([edges(im) for im in E])
    ref = E.mean(0) > 0.6
    stable = [round(float((e & ref).sum() / max(ref.sum(), 1)), 2) for e in E]
    return {"fps": round(fps, 3), "size": size, "frames": n, "minutes": round(n / fps / 60, 1),
            "stable": stable, "stable_min": min(stable) if stable else None}, sheet


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "data", "cctv", "*", "*.mp4")))
    os.makedirs(OUT, exist_ok=True)
    for p in paths:
        tag = os.path.basename(os.path.dirname(p)) + "__" + os.path.splitext(os.path.basename(p))[0]
        r = probe(p)
        info, sheet = r if isinstance(r, tuple) else (r, None)
        json.dump(info, open(os.path.join(OUT, tag + ".json"), "w"))
        if sheet is not None:
            cv2.imwrite(os.path.join(OUT, tag + ".jpg"), sheet)
        print(tag, {k: v for k, v in info.items() if k != "stable"}, flush=True)


if __name__ == "__main__":
    main()
