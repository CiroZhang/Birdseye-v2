"""Broadcast audio of every CCTV rally, cut from the whole-day match video on the TV clip's clock.

The arena sound is one mix shared by all four feeds (a racket hit is the same pop in each, to a few ms), so the TV
feed's audio is used, starting at the TV clip's first frame: audio time t s = TV frame 25 t.
Output outputs/cctv/audio/<uid>.npy  int16 mono at 16 kHz.
usage: python tools/cctv_audio.py [uid ...]   (default: every rally of every session under data/cctv/*/rallies)"""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 16000
OUT = os.path.join(ROOT, "outputs", "cctv", "audio")


def cut(meta_path):
    uid = os.path.basename(os.path.dirname(meta_path)); dst = os.path.join(OUT, uid + ".npy")
    if os.path.exists(dst):
        return uid, "skip"
    m = json.load(open(meta_path)); sess = os.path.dirname(os.path.dirname(os.path.dirname(meta_path)))
    src = os.path.join(sess, m["views"]["tv"] + ".mp4")
    raw = subprocess.run([FF, "-v", "error", "-ss", f"{m['tv_start_s']:.3f}", "-t", f"{m['len_s']:.3f}", "-i", src,
                          "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.int16)
    if len(a) < SR:
        return uid, "too short"
    np.save(dst, a)
    return uid, "ok"


def main():
    os.makedirs(OUT, exist_ok=True)
    metas = sorted(glob.glob(os.path.join(ROOT, "data", "cctv", "*", "rallies", "*", "meta.json")))
    if sys.argv[1:]:
        metas = [m for m in metas if os.path.basename(os.path.dirname(m)) in set(sys.argv[1:])]
    with ThreadPoolExecutor(8) as ex:
        for i, (uid, st) in enumerate(ex.map(cut, metas)):
            if st != "skip" and (i % 100 == 0 or st != "ok"):
                print(i, uid, st, flush=True)
    print("DONE", len(metas), flush=True)


if __name__ == "__main__":
    main()
