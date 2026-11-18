"""Round-3 labels for the video hit model (self-training, second round).
Positives: physics-verified teacher hits, plus peaks where the round-2 video model and the sound+vision model agree
(video >= V_HI, or video >= V_MID with a sound+vision peak >= F_MID within 3 frames), >= 4 frames from each other.
Negatives: confident mid-flight frames as before, plus frames where BOTH models are sure there is no hit (video and
sound+vision < LO for +-3 frames) anywhere in the clip -- pre-serve, after the point, between shots -- > 8 frames from
every positive. Side: verified hits keep the teacher's side; new ones take the round-2 video side when it is sure.
usage: python tools/hit_labels3.py <round-2 per-frame probs dir>"""
import json, os, sys
import importlib.util
import numpy as np, torch
from scipy.signal import find_peaks

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V_HI, V_MID, F_MID, LO = 0.85, 0.5, 0.5, 0.1
os.environ["AH_MODE"] = "av"
spec = importlib.util.spec_from_file_location("ah", os.path.join(ROOT, "tools", "audio_hits.py"))
AH = importlib.util.module_from_spec(spec); spec.loader.exec_module(AH)
net = AH.Net().to(AH.DEV); net.load_state_dict(torch.load(os.path.join(AH.OUT, "model.pt"))); net.eval()
vdir = sys.argv[1]
lab = json.load(open(os.path.join(ROOT, "outputs", "cctv", "hitclips", "labels.json")))
rng = np.random.default_rng(3); n_new = n_neg = 0
for i, (u, d) in enumerate(lab.items()):
    vp = os.path.join(vdir, u + ".npz")
    if "error" in d or not os.path.exists(vp):
        continue
    z = np.load(vp); vh = np.convolve(z["hit"], [.25, .5, .25], "same"); far = z["far"]; n = len(vh)
    pa = AH.prob(net, AH.feats(u))                       # 200 Hz -> per TV frame max
    fh = np.array([pa[f * 8:(f + 1) * 8].max() if f * 8 < len(pa) else 0.0 for f in range(n)])
    pos = [list(p) for p in d["pos"]]
    have = [h for h, _ in pos]
    for f in find_peaks(vh, height=V_MID, distance=5)[0]:
        a, b = max(0, f - 3), min(n, f + 4)
        if vh[f] >= V_HI or fh[a:b].max() >= F_MID:
            if all(abs(f - h) >= 4 for h in have):
                side = 1 if far[f] > 0.9 else 0 if far[f] < 0.1 else -1
                pos.append([float(f), side]); have.append(float(f)); n_new += 1
    allp = np.array(have) if have else np.zeros(0)
    far_from = lambda c: not len(allp) or np.abs(allp - c).min() > 8
    neg = [c for c in d["neg"] if far_from(c)]
    quiet = [c for c in range(3, n - 3) if vh[c - 3:c + 4].max() < LO and fh[c - 3:c + 4].max() < LO and far_from(c)]
    extra = list(rng.choice(quiet, size=min(20, len(quiet)), replace=False)) if quiet else []
    d["pos"] = pos; d["neg"] = sorted(set(neg) | {int(c) for c in extra}); n_neg += len(extra)
    if i % 200 == 0:
        print(i, u, "pos", len(pos), "neg", len(d["neg"]), flush=True)
json.dump(lab, open(os.path.join(ROOT, "outputs", "cctv", "hitclips", "labels3.json"), "w"))
print("DONE new positives", n_new, "quiet negatives", n_neg, "total positives", sum(len(d.get("pos", [])) for d in lab.values()))
