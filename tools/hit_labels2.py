"""Round-2 labels for the video hit model (self-training): positives = physics-verified teacher hits PLUS confident
peaks of the sound+vision hit model (tools/audio_hits.py AH_MODE=av, prob >= THR, >= 4 frames from any positive);
negatives = confident mid-flight frames > 8 frames from every positive. Side is kept only for verified hits (-1 =
unknown for the new ones). Reads/writes outputs/cctv/hitclips/labels.json -> labels2.json.
usage: python tools/hit_labels2.py [thr]"""
import json, os, sys
import importlib.util
import numpy as np, torch
from scipy.signal import find_peaks

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
os.environ["AH_MODE"] = "av"
spec = importlib.util.spec_from_file_location("ah", os.path.join(ROOT, "tools", "audio_hits.py"))
AH = importlib.util.module_from_spec(spec); spec.loader.exec_module(AH)
net = AH.Net().to(AH.DEV); net.load_state_dict(torch.load(os.path.join(AH.OUT, "model.pt"))); net.eval()
lab = json.load(open(os.path.join(ROOT, "outputs", "cctv", "hitclips", "labels.json")))
added = 0
for i, (u, d) in enumerate(lab.items()):
    if "error" in d:
        continue
    try:
        p = AH.prob(net, AH.feats(u))
    except Exception as e:
        print(u, "skip", e); continue
    pk = find_peaks(np.convolve(p, np.ones(5) / 5, "same"), height=THR, distance=int(0.2 * AH.AF))[0] / AH.AF * 25
    have = [h for h, _ in d["pos"]]
    new = [float(f) for f in pk if f < d["n"] and all(abs(f - h) >= 4 for h in have)]
    d["pos"] = d["pos"] + [[f, -1] for f in new]; added += len(new)
    allp = np.array([h for h, _ in d["pos"]]) if d["pos"] else np.zeros(0)
    d["neg"] = [c for c in d["neg"] if not len(allp) or np.abs(allp - c).min() > 8]
    if i % 200 == 0:
        print(i, u, "added", len(new), flush=True)
json.dump(lab, open(os.path.join(ROOT, "outputs", "cctv", "hitclips", "labels2.json"), "w"))
print("DONE added", added, "positives; total", sum(len(d.get("pos", [])) for d in lab.values()))
