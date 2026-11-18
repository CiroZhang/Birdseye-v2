"""One pass over a whole-day CCTV feed: 5 fps grey thumbnails (96x54) and a 100 Hz audio onset envelope, the raw
material for finding the rallies (tools/cctv_rallies.py) and syncing the views by sound.

Output: <video>.thumbs.npy (N,54,96) uint8 at 5 fps, <video>.onset.npy float32 at 100 Hz, next to the video
usage: python tools/cctv_scan.py <video.mp4> [...]   (one ffmpeg per video, run in parallel)"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg
import numpy as np

FF = imageio_ffmpeg.get_ffmpeg_exe()
TW, TH, TFPS = 96, 54, 5
SR = 8000


def thumbs(src):
    dst = src[:-4] + ".thumbs.npy"
    if os.path.exists(dst):
        return
    p = subprocess.Popen([FF, "-v", "error", "-i", src, "-an", "-vf", f"fps={TFPS},scale={TW}:{TH}:flags=area",
                          "-pix_fmt", "gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    raw = p.stdout.read()
    p.wait()
    np.save(dst, np.frombuffer(raw, np.uint8).reshape(-1, TH, TW))
    print(os.path.basename(dst), len(raw) // (TW * TH), "frames", flush=True)


def onset(src):
    dst = src[:-4] + ".onset.npy"
    if os.path.exists(dst):
        return
    p = subprocess.Popen([FF, "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         stdout=subprocess.PIPE)
    env, rest = [], np.zeros(0, np.float32)
    while True:
        buf = p.stdout.read(SR * 2 * 60)
        if not buf:
            break
        x = np.concatenate([rest, np.frombuffer(buf, np.int16).astype(np.float32)])
        d = np.abs(np.diff(x))
        k = len(d) // 80 * 80
        env.append(d[:k].reshape(-1, 80).mean(1))
        rest = x[k:]
    p.wait()
    np.save(dst, np.concatenate(env).astype(np.float32))
    print(os.path.basename(dst), "done", flush=True)


if __name__ == "__main__":
    jobs = [(f, s) for s in sys.argv[1:] for f in (thumbs, onset)]
    with ThreadPoolExecutor(len(jobs)) as ex:
        list(ex.map(lambda j: j[0](j[1]), jobs))
