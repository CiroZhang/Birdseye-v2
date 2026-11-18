"""Shared helpers for the runtime benchmark (all methods, one L40S node).

- uids(): warm-up uid + the first 20 CCTV test rallies (outputs/cctv/split.json "test").
- predecode(paths): decode every clip into memory with the real cv2.VideoCapture (NOT timed), then
  install MemCap as cv2.VideoCapture so every existing tool that opens the clip by path reads the cached
  frames instead of decoding (frames are read-only so no tool can mutate the cache).
- Timer: per-stage wall time with torch.cuda.synchronize() before and after.
"""
import json, os, platform, subprocess, sys, time

import cv2
import numpy as np

TIMING = "/scratch/ciro/timing"
ROOT = os.path.join(TIMING, "root")
CLIPS = os.path.join(ROOT, "data", "cctv", "clips")
FPS = 25.0
N_TEST = int(os.environ.get("N_TEST", "20"))


def uids():
    test = json.load(open(os.path.join(ROOT, "outputs", "cctv", "split.json")))["test"]
    return test[N_TEST], test[:N_TEST]  # warm-up = the 21st test rally (held out, not counted)


def clip(uid):
    return os.path.join(CLIPS, uid + ".mp4")


_RealCap = cv2.VideoCapture
CACHE = {}  # abs path -> (frames list, fps, (w, h))


def decode(path):
    cap = _RealCap(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fr = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        im.setflags(write=False)
        fr.append(im)
    cap.release()
    h, w = fr[0].shape[:2]
    return fr, fps, (w, h)


class MemCap:
    """cv2.VideoCapture over pre-decoded frames (read / grab / retrieve / get / set / isOpened / release)."""

    def __init__(self, path, *a):
        self.fr, self.fps, (self.w, self.h) = CACHE[os.path.abspath(path)]
        self.pos, self.opened, self.last = 0, True, None

    def isOpened(self):
        return self.opened

    def grab(self):
        if self.pos >= len(self.fr):
            self.last = None
            return False
        self.last = self.fr[self.pos]
        self.pos += 1
        return True

    def retrieve(self, *a):
        return (self.last is not None), self.last

    def read(self, *a):
        return (True, self.last) if self.grab() else (False, None)

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_COUNT: float(len(self.fr)), cv2.CAP_PROP_FPS: float(self.fps),
                cv2.CAP_PROP_FRAME_WIDTH: float(self.w), cv2.CAP_PROP_FRAME_HEIGHT: float(self.h),
                cv2.CAP_PROP_POS_FRAMES: float(self.pos)}.get(prop, 0.0)

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(max(0, min(int(val), len(self.fr))))
            return True
        return False

    def release(self):
        self.opened = False


def _cap(path, *a):
    p = os.path.abspath(path) if isinstance(path, str) else path
    if isinstance(p, str) and p in CACHE:
        return MemCap(p)
    return _RealCap(path, *a)


def predecode(paths):
    for p in paths:
        CACHE[os.path.abspath(p)] = decode(p)
    cv2.VideoCapture = _cap


def drop(path):
    CACHE.pop(os.path.abspath(path), None)


def n_frames(path):
    return len(CACHE[os.path.abspath(path)][0])


def sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        tm = self

        class _C:
            def __enter__(s):
                sync()
                s.t0 = time.perf_counter()

            def __exit__(s, *exc):
                sync()
                tm.t[name] = tm.t.get(name, 0.0) + time.perf_counter() - s.t0
        return _C()


def hw_info():
    info = {"python": sys.version.split()[0], "host": platform.node()}
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()
    except Exception as e:
        info["gpu"] = repr(e)
    try:
        info["cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        info["cpu"] = platform.processor()
    info["slurm_cpus"] = os.environ.get("SLURM_CPUS_PER_TASK")
    info["slurm_job"] = os.environ.get("SLURM_JOB_ID")
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    torch_gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    return info


def dump(name, obj):
    os.makedirs(os.path.join(TIMING, "out"), exist_ok=True)
    p = os.path.join(TIMING, "out", name + ".json")
    json.dump(obj, open(p, "w"), indent=1)
    print("wrote", p, flush=True)
