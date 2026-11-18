"""Review page for BirdsCourt calibration problems (tools/fb_birdscourt.py), FineBadminton-20K train + FBBench.

Groups: still failed (fallback camera), fixed by the 10-frame retry, left/right swapped by BirdsCourt (relabelled).
Images are downscaled and embedded as data URIs in outputs/calib_review/data.json (one file to publish)
usage: python tools/export_calib_review.py"""
import base64, glob, json, os

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DST = os.path.join(ROOT, "outputs", "calib_review")
SETS = {"train": ("fb20k", "FineBadminton-20K rally"), "bench": ("fb", "FBBench clip")}
THUMB_W = 480


def thumb(src, name, w=THUMB_W):
    if not os.path.exists(src):
        return None
    im = cv2.imread(src)
    if im is None:
        return None
    im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 72])[1]).decode()


def main():
    rows, counts = [], {}
    for key, (out, label) in SETS.items():
        base = os.path.join(ROOT, "outputs", out)
        cals = sorted(glob.glob(os.path.join(base, "calib_bc", "*.json")))
        n_valid = n_swap = 0
        for p in cals:
            uid = os.path.basename(p)[:-5]
            st = json.load(open(p))["stats"]
            n_valid += bool(st.get("valid")); n_swap += bool(st.get("lr_swapped"))
            retried = os.path.exists(os.path.join(base, "court_pts", uid + ".median_failed.json"))
            if st.get("valid") and not retried:
                continue
            pts = json.load(open(os.path.join(base, "court_pts", uid + ".json"))) if os.path.exists(os.path.join(base, "court_pts", uid + ".json")) else {}
            if st.get("valid"):
                group, why = "fixed", f"median image failed; {st['retry']['frames_used']} of {st['retry']['frames_tried']} single frames agreed"
            else:
                group = "failed"
                why = st.get("reason") or pts.get("error") or (f"fit error {st.get('rms_px')} px" if "rms_px" in st else "unknown")
                if "rms_px" in st and st.get("rms_px", 0) < 4:
                    why = f"camera fit implausible (centre {st.get('center')})"
            med = os.path.join(base, "court_img", uid + (".median_failed.jpg" if retried else ".jpg"))
            retry_frames = sorted(glob.glob(os.path.join(base, "court_img_retry", uid + "_*.jpg")))
            rows.append({
                "set": key, "set_label": label, "uid": uid, "group": group, "why": why[:160],
                "median": thumb(med, f"{key}_{uid}_median.jpg"),
                "fit": thumb(os.path.join(base, "calib_bc", uid + ".jpg"), f"{key}_{uid}_fit.jpg") if st.get("valid") else None,
                "frames": [t for t in (thumb(f, f"{key}_{os.path.basename(f)}", 240) for f in retry_frames[::3][:4]) if t],
            })
        counts[key] = {"label": label, "clips": len(cals), "valid": n_valid, "lr_swapped": n_swap}
    os.makedirs(DST, exist_ok=True)
    json.dump({"counts": counts, "rows": rows}, open(os.path.join(DST, "data.json"), "w"), indent=0)
    for k, c in counts.items():
        print(k, c, "failed", sum(r["group"] == "failed" and r["set"] == k for r in rows),
              "fixed", sum(r["group"] == "fixed" and r["set"] == k for r in rows))


if __name__ == "__main__":
    main()
