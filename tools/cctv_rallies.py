"""Find the rallies of one CCTV session (all of its views downloaded and scanned by tools/cctv_scan.py) and cut each
from every view, synced, into its own folder. The original whole-day videos are left untouched.

  sync    per-minute audio cross-correlation of each view's onset envelope against the TV feed (2 min windows,
          +-15 s search), median-filtered; the feeds drift ~0.1 s over the day and the TV feed drops ~6 s once
  rallies the TV feed's main (behind-the-court) camera: each frame's gradient image correlated with the median of
          a known main-camera stretch; runs above 0.7, gaps < 1 s merged, >= 4 s kept (close-ups, replays and
          crowd shots score < 0.5)

Output: <session>/rallies/<session>_<k>/{tv,side,p1,p2}.mp4 + meta.json {tv_start_s, len_s, offsets_s},
        <session>/rallies/index.json
usage: python tools/cctv_rallies.py <session_dir> tv=<id> side=<id> p1=<id> p2=<id> [--ref <tv seconds of main cam>]
       (no --ref: the main camera is found as the day's most frequent shot)"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve

FF = imageio_ffmpeg.get_ffmpeg_exe()
TFPS = 5
W_SYNC, M_SYNC, STEP = 12000, 1500, 6000  # 100 Hz samples: 2 min window, +-15 s search, every minute
ON, MERGE, MIN_LEN = 0.7, 5, 20  # main-camera score threshold; gap / length in 5 fps frames


def nz(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def sync(tv, other):
    out = []
    for a in range(M_SYNC, len(tv) - W_SYNC - M_SYNC, STEP):
        c = fftconvolve(nz(other[a - M_SYNC:a + W_SYNC + M_SYNC]), nz(tv[a:a + W_SYNC])[::-1], "valid")
        out.append(((a + W_SYNC / 2) / 100, (c.argmax() - M_SYNC) / 100, c.max() / W_SYNC))
    o = np.array(out)
    return o[:, 0], median_filter(o[:, 1], 5), o[:, 2]


def grad(x):
    x = x.astype(np.float32)
    return np.sqrt(cv2.Sobel(x, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(x, cv2.CV_32F, 0, 1) ** 2)


def auto_ref(thumbs, n=1500):
    """The main camera's look without being told where it is: of n frames spread over the day, the one with the most
    near-identical others (it is by far the most frequent shot); the median of that group's gradient images."""
    idx = np.linspace(0, len(thumbs) - 1, n).astype(int)
    g = np.stack([grad(thumbs[k]) for k in idx]).reshape(n, -1)
    g = (g - g.mean(1, keepdims=True)) / (g.std(1, keepdims=True) + 1e-6)
    c = g @ g.T / g.shape[1]
    k = int(np.argmax((c > ON).sum(1)))
    grp = idx[c[k] > ON]
    print(f"main camera: {len(grp)} of {n} sampled frames, e.g. t = {idx[k] / TFPS:.0f} s", flush=True)
    return nz(np.median(np.stack([grad(thumbs[i]) for i in grp[:200]]), 0))


def main_cam(thumbs, ref_s=None):
    if ref_s is None:
        ref = auto_ref(thumbs)
    else:
        i = int(ref_s * TFPS)
        ref = nz(np.median(np.stack([grad(thumbs[k]) for k in range(i - 50, i + 50, 5)]), 0))
    sc = np.empty(len(thumbs), np.float32)
    for a in range(0, len(thumbs), 5000):
        g = np.stack([grad(b) for b in thumbs[a:a + 5000]])
        g = (g - g.mean((1, 2), keepdims=True)) / (g.std((1, 2), keepdims=True) + 1e-6)
        sc[a:a + 5000] = (g * ref).mean((1, 2))
    d = np.diff(np.r_[0, (sc > ON).astype(int), 0])
    seg = []
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if seg and a - seg[-1][1] < MERGE:
            seg[-1][1] = b
        else:
            seg.append([a, b])
    return [(a / TFPS, (b - a) / TFPS) for a, b in seg if b - a >= MIN_LEN]


def cut(src, start, dur, dst):
    if os.path.exists(dst):
        return
    subprocess.run([FF, "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src, "-an",
                    "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", "-r", "25", dst + ".tmp.mp4"], check=True)
    os.replace(dst + ".tmp.mp4", dst)


def main():
    sess = sys.argv[1]
    views = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    ref_s = float(sys.argv[sys.argv.index("--ref") + 1]) if "--ref" in sys.argv else None
    name = os.path.basename(os.path.normpath(sess))
    on = {v: np.load(os.path.join(sess, i + ".onset.npy")) for v, i in views.items()}
    offs = {v: sync(on["tv"], on[v]) for v in views if v != "tv"}
    segs = main_cam(np.load(os.path.join(sess, views["tv"] + ".thumbs.npy"), mmap_mode="r"), ref_s)
    out = os.path.join(sess, "rallies")
    os.makedirs(out, exist_ok=True)
    jobs, index = [], []
    for k, (t0, dur) in enumerate(segs):
        uid = f"{name}_{k:04d}"
        d = os.path.join(out, uid)
        os.makedirs(d, exist_ok=True)
        o = {v: float(np.interp(t0 + dur / 2, *offs[v][:2])) for v in offs}
        o["tv"] = 0.0
        json.dump({"tv_start_s": t0, "len_s": dur, "offsets_s": o, "views": views}, open(os.path.join(d, "meta.json"), "w"))
        index.append({"uid": uid, "tv_start_s": t0, "len_s": dur})
        jobs += [(os.path.join(sess, views[v] + ".mp4"), t0 + o[v], dur, os.path.join(d, v + ".mp4")) for v in views]
    json.dump({"session": name, "views": views, "rallies": index}, open(os.path.join(out, "index.json"), "w"), indent=1)
    print(name, len(segs), "rallies,", round(sum(d for _, d in segs) / 60, 1), "min", flush=True)
    done = [0]

    def run(j):
        cut(*j)
        done[0] += 1
        if done[0] % 200 == 0:
            print(done[0], "/", len(jobs), "clips", flush=True)

    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "8"))) as ex:
        list(ex.map(run, jobs))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
