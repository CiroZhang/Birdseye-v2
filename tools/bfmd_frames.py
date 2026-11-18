"""Every frame of the BFMD rallies in KEEP, resized to SIZE x SIZE (aspect squashed, as tools/bfmd_rgb_stroke.py), into
one uint8 memmap, for the RGB input of tools/stroke3d.py (INPUT=rgb).

out  data/bfmd/frames<SIZE>/frames.u8 (N, SIZE, SIZE, 3) and index.json {tag: [offset, n_frames]}
usage: python tools/bfmd_frames.py [n_workers]   (env SIZE=128, KEEP=outputs/stroke_keep_bfmd.txt)"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIZE = int(os.environ.get("SIZE", "128"))
SET = os.environ.get("SET", "bfmd")  # fbs: FineBadminton-20K clips (data/fb20k/clips/<match>_<KKK>.mp4)
KEEP = os.environ.get("KEEP", os.path.join(ROOT, "outputs", f"stroke_keep_{SET}.txt"))
CLIPS = os.path.join(ROOT, "data", *(("bfmd", "clips") if SET == "bfmd" else ("fb20k", "clips")))
OUT = os.path.join(ROOT, "data", "bfmd" if SET == "bfmd" else "fbs", f"frames{SIZE}")


def clip(tag):
    if SET == "bfmd":
        return os.path.join(CLIPS, tag.replace("::", "__") + ".mp4")
    m, rt = tag.split("::")
    return os.path.join(CLIPS, f"{m}_{int(rt.split('_r')[1]):03d}.mp4")


def count(tag):
    c = cv2.VideoCapture(clip(tag))
    return tag, int(c.get(cv2.CAP_PROP_FRAME_COUNT))


def decode(args):
    tag, off, n = args
    mm = np.memmap(os.path.join(OUT, "frames.u8"), np.uint8, "r+", shape=(off + n, SIZE, SIZE, 3))
    c = cv2.VideoCapture(clip(tag))
    k, last = 0, np.zeros((SIZE, SIZE, 3), np.uint8)
    while k < n:
        ok, im = c.read()
        if ok:
            last = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        mm[off + k] = last  # a short decode repeats the last frame
        k += 1
    mm.flush()
    return n


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    tags = sorted(open(KEEP).read().split())
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(nw) as ex:
        counts = dict(ex.map(count, tags))
    index, off = {}, 0
    for t in tags:
        index[t] = [off, counts[t]]
        off += counts[t]
    np.memmap(os.path.join(OUT, "frames.u8"), np.uint8, "w+", shape=(off, SIZE, SIZE, 3)).flush()
    json.dump({"size": SIZE, "n": off, "index": index}, open(os.path.join(OUT, "index.json"), "w"))
    print(f"{len(tags)} rallies, {off} frames, {off * SIZE * SIZE * 3 / 1e9:.1f} GB", flush=True)
    done = 0
    with ProcessPoolExecutor(nw) as ex:
        for i, n in enumerate(ex.map(decode, [(t, *index[t]) for t in tags], chunksize=4)):
            done += n
            if i % 100 == 0:
                print(f"{i}/{len(tags)} rallies, {done} frames", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
