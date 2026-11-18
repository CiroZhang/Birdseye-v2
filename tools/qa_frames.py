"""Frames for the video (RGB) condition of the QA experiments: every clip decoded once at FPS frames per second (at most
MAX frames, uniformly spaced beyond that), resized to WIDTH px wide, JPEG-encoded, with their times in the clip.

out  outputs/qa_frames/<set>/<uid>.npz  {jpg: object array of JPEG bytes, t: (N,) seconds}
usage: python tools/qa_frames.py fb|fb20k [n_workers]"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VID = {"fb": os.path.join(ROOT, "data", "fbbench", "hf_video_clips_qa"), "fb20k": os.path.join(ROOT, "data", "fb20k", "clips")}
FPS, MAX, WIDTH = 4.0, 160, 448


def one(args):
    st, path = args
    uid = os.path.basename(path)[:-4]
    dst = os.path.join(ROOT, "outputs", "qa_frames", st, uid + ".npz")
    if os.path.exists(dst):
        return 0
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    want = np.arange(0, n / fps, 1 / FPS)
    if len(want) > MAX:
        want = np.linspace(0, n / fps - 1 / fps, MAX)
    idx = set(np.minimum(np.round(want * fps).astype(int), max(n - 1, 0)).tolist())
    jpg, t = [], []
    k = 0
    while True:
        ok = c.grab()
        if not ok:
            break
        if k in idx:
            ok, im = c.retrieve()
            if ok:
                h = int(round(im.shape[0] * WIDTH / im.shape[1] / 28) * 28)  # Qwen2.5-VL patch grid
                im = cv2.resize(im, (WIDTH, h), interpolation=cv2.INTER_AREA)
                jpg.append(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
                t.append(k / fps)
        k += 1
    arr = np.empty(len(jpg), object)
    arr[:] = jpg
    np.savez(dst, jpg=arr, t=np.array(t, np.float32))
    return len(jpg)


def main():
    st = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    os.makedirs(os.path.join(ROOT, "outputs", "qa_frames", st), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(VID[st], "*.mp4")))
    tot = 0
    with ProcessPoolExecutor(nw) as ex:
        for i, n in enumerate(ex.map(one, [(st, p) for p in paths], chunksize=4)):
            tot += n
            if i % 200 == 0:
                print(f"{i}/{len(paths)} frames so far {tot}", flush=True)
    print(f"done {st}: {len(paths)} clips, {tot} frames", flush=True)


if __name__ == "__main__":
    main()
