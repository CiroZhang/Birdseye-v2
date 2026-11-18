"""FineBadminton's stroke labels from our BFMD stroke classifier, learned on FineBadminton-20K training rallies
(non-benchmark matches only): the two label sets cut strokes differently (FineBadminton's "push shot" is its most
common hit and covers much of what BFMD calls net shots / drives / lifts), so the hand mapping in
fbbench_answer.STROKE is weak.

1. FB transitions: P(next hit type | hit type) counted from every training rally's labels.
2. Label map: run the stroke ensemble on the reconstructed training rallies that have a shuttle track, take the
   hitter's class probabilities at each labelled hit, fit a multinomial logistic regression -> FB type.
   Evaluated leave-matches-out against the hand mapping before the final fit on everything.

Rallies at 25 fps are built at their native rate (bodies, shuttle rays), then resampled to 30 fps -- the rate the
model was trained at -- with velocities recomputed.
Output: outputs/fb20k/label_map.json  {fb_types, bfmd_vocab, coef, intercept, transitions, cv}
usage: CUDA_VISIBLE_DEVICES=-1 python tools/fb_label_map.py"""
import glob, json, os, sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fbbench_answer as A  # noqa: E402

B, tf = A.B, A.tf
CLIPS = os.path.join(ROOT, "data", "fb20k", "clips")
FB20 = os.path.join(ROOT, "outputs", "fb20k")
OUT = os.path.join(FB20, "label_map.json")
# FineBadminton-20K says "kill" where the benchmark options say "smash"
FB_TYPES = ["push shot", "smash", "block", "net shot", "serve", "drive", "drop shot", "clear", "cross-court net shot",
            "net lift", "net kill"]
RENAME = {"kill": "smash"}
WIN = 5  # frames: the hitter's hit-probability peak is looked for this close to the labelled frame


def fb_type(h):
    t = RENAME.get(h["type"], h["type"])
    return t if t in FB_TYPES else None


ZONES = ["left front", "mid front", "right front", "left mid", "mid", "right mid", "left back", "mid back", "right back"]


def zone_of(area):
    z = area.replace(" court", "").strip()
    return ZONES.index(z) if z in ZONES else None


def transitions():
    """-> next-type counts (type x type), next hit's zone counts (type x zone)."""
    C = np.zeros((len(FB_TYPES), len(FB_TYPES)))
    Z = np.zeros((len(FB_TYPES), len(ZONES)))
    for p in glob.glob(os.path.join(CLIPS, "*.json")):
        hits = json.load(open(p, encoding="utf8"))["hits"]
        for a, b in zip(hits, hits[1:]):
            ta, tb = fb_type(a), fb_type(b)
            if ta and tb:
                C[FB_TYPES.index(ta), FB_TYPES.index(tb)] += 1
            if ta and zone_of(b["area"]) is not None:
                Z[FB_TYPES.index(ta), zone_of(b["area"])] += 1
    return C, Z


def resample(r, fps):
    """A native-rate rally (as fbbench_answer.build + set_pelvis/add_shuttle leave it) -> 30 fps, in place."""
    n = r["n_frames"]
    n30 = int(n * 30 / fps)
    idx = np.minimum(np.round(np.arange(n30) * fps / 30).astype(int), n - 1)
    for s in ("top", "bottom"):
        rel = np.asarray(r[s + "_rel"], np.float32)[idx]
        v = np.zeros_like(rel)
        v[1:] = rel[1:] - rel[:-1]
        v[~np.isfinite(v)] = 0
        r[s + "_rel"], r[s + "_vel"] = rel, v
        r["_world"][s] = r["_world"][s][idx]
    r["n_frames"] = n30


def build(uids):
    """fbbench_answer.build for FineBadminton-20K clips: {uid: (rally at 30 fps, fps)}."""
    sd = os.path.join(FB20, "sd")
    rallies, joints, fpss = {}, {}, {}
    for uid in uids:
        d = np.load(os.path.join(FB20, "vimo_bc", uid + ".npz"))
        cap = cv2.VideoCapture(os.path.join(CLIPS, uid + ".mp4"))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        J = {s: A.to_30fps(d[s], d["frames"], n) for s in ("top", "bottom")}
        joints[uid] = {"start": 0, "end": n - 1, "joints": {s: J[s].tolist() for s in J}}
        rel = {s: J[s] - J[s][:, :1] for s in J}
        rel["top"][..., :2] *= -1
        r = {"n_frames": n, "_world": J}
        for s in J:
            v = np.zeros_like(rel[s])
            v[1:] = rel[s][1:] - rel[s][:-1]
            r[s + "_rel"], r[s + "_vel"] = rel[s], v
        rallies["fb20k::" + uid] = r
        fpss[uid] = fps
    os.makedirs(os.path.join(sd, "fb20k"), exist_ok=True)
    json.dump({"rallies": joints}, open(os.path.join(sd, "fb20k", "joints_selfdetected.json"), "w"))
    B.SRC.update(sd=sd, out=FB20, uid=lambda m, rt: rt)
    dataset = {"rallies": rallies}
    B.set_pelvis(dataset)
    if "shuttle" in B.FEATS:
        B.add_shuttle(dataset)
    if "racket" in B.FEATS:
        B.add_racket(dataset)
        tf.LR_PERM = tf.LR_PERM + [len(tf.LR_PERM)]
    out = {}
    for uid in uids:
        r = rallies["fb20k::" + uid]
        if abs(fpss[uid] - 30) > 0.5:
            resample(r, fpss[uid])
        out[uid] = (r, fpss[uid])
    return out


def examples(uids):
    """-> X (log class probabilities at each labelled hit), y (FB type index), match ids, and (zone index, lat, dep)
    of the hitter's reconstructed position at each labelled hit."""
    data = build(uids)
    vocab = json.load(open(f"{tf.BASE}/training_dataset_all_selfdetected.json"))["shot_type_vocab"]
    E, _ = A.load_model_for(next(iter(data.values()))[0], vocab)
    X, y, g, pos = [], [], [], []
    for k, (uid, (r, fps)) in enumerate(data.items()):
        cv = A.curves_for(E, r)
        for h in json.load(open(os.path.join(CLIPS, uid + ".json"), encoding="utf8"))["hits"]:
            t = fb_type(h)
            if t is None or h["hitter"] not in cv:
                continue
            probs, sp = cv[h["hitter"]]
            f = int(round(h["frame"] * 30 / fps))
            lo, hi = max(0, f - WIN), min(len(probs), f + WIN + 1)
            if lo >= hi:
                continue
            f = lo + int(np.argmax(probs[lo:hi]))
            P, z = A.position(r, h["hitter"], f), zone_of(h["area"])
            if P is not None and z is not None:
                pos.append((z, *A.own_frame(P, h["hitter"])))
            X.append(np.log(sp[f] + 1e-4)); y.append(FB_TYPES.index(t)); g.append(uid.split("_")[0])
        if k % 25 == 0:
            print(f"{k}/{len(data)} rallies, {len(y)} hits", flush=True)
    return np.array(X), np.array(y), np.array(g), vocab, np.array(pos, float)


def hand_map(X, vocab):
    """fbbench_answer.STROKE as a classifier (the baseline); serve -> its own BFMD class when there is one."""
    p = np.exp(X)
    sc = np.zeros((len(X), len(FB_TYPES)))
    for j, t in enumerate(FB_TYPES):
        cls = A.STROKE.get(t, ["short_service", "long_service"] if t == "serve" else [])
        sc[:, j] = sum(p[:, vocab.index(c)] for c in cls if c in vocab) if cls else 0
    return sc.argmax(1)


def main():
    from sklearn.linear_model import LogisticRegression
    C, Z = transitions()
    have = lambda u: (os.path.exists(os.path.join(FB20, "vimo_bc", u + ".npz")) and os.path.exists(os.path.join(FB20, "refined", u + ".json"))
                      and json.load(open(os.path.join(FB20, "calib_bc", u + ".json"))).get("cam"))
    uids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(FB20, "refined", "*.json")))
    uids = [u for u in uids if have(u)]
    print(len(uids), "rallies with bodies, shuttle and camera", flush=True)
    X, y, g, vocab, pos = examples(uids)
    zc = {ZONES[k]: np.median(pos[pos[:, 0] == k, 1:], 0).round(2).tolist() for k in range(len(ZONES)) if (pos[:, 0] == k).any()}
    print("zone centres (lat, dep) of our hitters:", zc, flush=True)
    print(len(y), "labelled hits", np.bincount(y, minlength=len(FB_TYPES)).tolist(), flush=True)
    ms = np.unique(g)
    folds = np.array_split(np.random.default_rng(0).permutation(ms), 5)
    acc_lr = acc_hand = 0
    for f in folds:
        te = np.isin(g, f)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[~te], y[~te])
        acc_lr += (lr.predict(X[te]) == y[te]).sum()
    acc_hand = (hand_map(X, vocab) == y).sum()
    cv = {"hits": int(len(y)), "matches": int(len(ms)), "lr_acc": acc_lr / len(y), "hand_acc": acc_hand / len(y),
          "majority_acc": float(np.bincount(y).max() / len(y))}
    print(cv, flush=True)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    coef = np.zeros((len(FB_TYPES), X.shape[1])); icpt = np.full(len(FB_TYPES), -30.0)
    coef[lr.classes_], icpt[lr.classes_] = lr.coef_, lr.intercept_  # a type never seen gets ~0 probability
    json.dump({"fb_types": FB_TYPES, "bfmd_vocab": vocab, "coef": coef.tolist(), "intercept": icpt.tolist(),
               "transitions": C.tolist(), "zones": ZONES, "next_zone": Z.tolist(), "zone_centres": zc, "cv": cv}, open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
