"""Hit labels for RallyNet's training rallies, where a dataset has them: {uid: {"top": [frames], "bottom": [frames]}}
(video frame index; top = the far player, y > NET_Y). BFMD's v1 test matches (train_final.load_split) are listed
separately and left out of training altogether, so the BFMD stroke/hit evaluation stays clean. DroneData and
FBBench have no training labels.

Output: outputs/rally_obs/hit_labels.json  {"labels": {set: {uid: {...}}}, "bfmd_test": [uid, ...]}"""
import glob, json, os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def bfmd():
    d = json.load(open(os.path.join(ROOT, "data", "bfmd", "v2", "training_dataset_all_selfdetected.json")))
    mor = d["match_of_rally"]
    matches = sorted(set(mor.values()))
    perm = np.random.default_rng(0).permutation(len(matches))
    test = {matches[i] for i in perm[:max(2, len(matches) // 5)]}  # v1 load_split
    lab, test_uids = {}, []
    for tag, r in d["rallies"].items():
        m, rt = tag.split("::")
        uid = f"{m}__{rt}"
        if mor[tag] in test:
            test_uids.append(uid)
            continue
        lab[uid] = {s: [int(i) for i, v in enumerate(r["target_" + s]) if float(v) >= 0.999] for s in ("top", "bottom")}
    return lab, test_uids


def fb20k():
    lab = {}
    for p in glob.glob(os.path.join(ROOT, "data", "fb20k", "clips", "*.json")):
        hits = json.load(open(p, encoding="utf8"))["hits"]
        lab[os.path.basename(p)[:-5]] = {s: [int(h["frame"]) for h in hits if h.get("hitter") == s] for s in ("top", "bottom")}
    return lab


if __name__ == "__main__":
    b, test = bfmd()
    out = {"labels": {"bfmd": b, "fb20k": fb20k()}, "bfmd_test": sorted(test)}
    json.dump(out, open(os.path.join(ROOT, "outputs", "rally_obs", "hit_labels.json"), "w"))
    n = lambda L: sum(len(v["top"]) + len(v["bottom"]) for v in L.values())
    print("bfmd", len(b), "rallies", n(b), "hits; test rallies excluded", len(test), "| fb20k", len(out["labels"]["fb20k"]), "rallies", n(out["labels"]["fb20k"]), "hits")
