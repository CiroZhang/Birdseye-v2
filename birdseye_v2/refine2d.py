"""Sub-pixel refinement of TrackNet shuttle detections.

TrackNet runs at 512x288; on 4K drone footage its peak lands anywhere along the shuttle's
motion-blur streak (front on one frame, tail on the next), i.e. up to ~0.2 m of jitter that
flips between shots. Here each detection is replaced by the centroid of the moving bright blob
around it (|frame - median background|), which is the mid-exposure position -- a consistent
time reference for both cameras."""
import cv2
import numpy as np


def _background(path, n=25):
    """Median of n frames spread over the clip (seeking; never holds the whole 4K clip)."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr = []
    for i in np.linspace(0, max(total - 2, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if ok:
            fr.append(im)
    return np.median(np.stack(fr), 0).astype(np.int16)


def refine_track(video_path, track, radius=60, diff_thr=40, min_area=6, max_area=4000):
    """track: (frame_idx (N,), px (N,2)). Streams the video once.
    Returns (px_refined (N,2), ok (N,) bool, area (N,)). ok=False when no plausible bright
    moving blob is near the detection (TrackNet point kept; callers treat it as unreliable)."""
    idx, px = track
    out, ok, areas = px.copy(), np.zeros(len(idx), bool), np.zeros(len(idx))
    if len(idx) == 0:
        return out, ok, areas
    bg = _background(video_path)
    h, w = bg.shape[:2]
    want = {int(f): k for k, f in enumerate(idx)}
    cap = cv2.VideoCapture(video_path)
    f = -1
    while True:
        ok_read, frame = cap.read()
        if not ok_read:
            break
        f += 1
        k = want.get(f)
        if k is None:
            continue
        x, y = px[k]
        x0, x1 = int(max(0, x - radius)), int(min(w, x + radius))
        y0, y1 = int(max(0, y - radius)), int(min(h, y + radius))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = frame[y0:y1, x0:x1].astype(np.int16)
        diff = np.abs(crop - bg[y0:y1, x0:x1]).sum(2)
        bright = crop.min(2) > 120  # shuttle is white: bright and unsaturated
        m = ((diff > diff_thr) & bright).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        best, bestd = None, 1e9
        for j in range(1, n):
            if not (min_area <= stats[j, cv2.CC_STAT_AREA] <= max_area):
                continue
            d = np.hypot(cent[j, 0] + x0 - x, cent[j, 1] + y0 - y)
            if d < bestd:
                best, bestd = j, d
        if best is not None and bestd < radius * 0.8:
            ys, xs = np.nonzero(lab == best)
            wgt = diff[ys, xs].astype(float)  # difference-weighted, blur-centred sub-pixel estimate
            out[k] = (x0 + (xs * wgt).sum() / wgt.sum(), y0 + (ys * wgt).sum() / wgt.sum())
            ok[k] = True
            areas[k] = stats[best, cv2.CC_STAT_AREA]
    return out, ok, areas
