"""Small helpers shared by the viewer exporters: rounding, normalised pixels, racket-face normals, preview clips."""
import cv2
import imageio_ffmpeg
import numpy as np

VW, VH = 960, 540


def r3(p):
    return None if p is None or not np.isfinite(p).all() else [round(float(v), 3) for v in p]


def norm_px(uv, size):
    return [round(float(uv[0]) / size[0], 4), round(float(uv[1]) / size[1], 4)]


def face_normals(w, u, min_speed=1.0, sigma=2.0):
    """Racket-face normal per frame: a swung racket moves face-first, so the normal is the head's velocity
    with its along-shaft part removed (head speed >= min_speed m/s); while the racket is (nearly) still it
    keeps the last swing's face; before any swing NaN (the viewer then faces the camera)."""
    from scipy.ndimage import gaussian_filter1d
    head = w + 0.48 * u
    v = np.gradient(np.nan_to_num(head), axis=0) * 30.0
    v = gaussian_filter1d(v, sigma, axis=0)
    vp = v - np.sum(v * u, 1, keepdims=True) * u
    sp = np.linalg.norm(vp, axis=1)
    n = np.full_like(u, np.nan)
    last = None
    for i in range(len(u)):
        if np.isfinite(u[i]).all() and sp[i] >= min_speed:
            last = vp[i] / sp[i]
        if last is not None and np.isfinite(u[i]).all():
            m = last - (last @ u[i]) * u[i]  # keep it perpendicular to the current shaft
            if np.linalg.norm(m) > 1e-3:
                n[i] = m / np.linalg.norm(m)
    return n


def write_clip(src, dst, f0, f1):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    w = imageio_ffmpeg.write_frames(dst, (VW, VH), fps=30, codec="libx264", quality=None,
                                    output_params=["-crf", "30", "-preset", "slow", "-movflags", "+faststart"],
                                    macro_block_size=1)
    w.send(None)
    for _ in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        w.send(np.ascontiguousarray(cv2.cvtColor(cv2.resize(img, (VW, VH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
    w.close()
    cap.release()
