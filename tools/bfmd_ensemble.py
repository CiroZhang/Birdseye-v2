"""Seed ensemble of tools/bfmd_stroke.py runs: average the hit probability and the stroke-type softmax of the
given checkpoints (same FEAT), pick the peak threshold on the clean selection matches (hit F1 + stroke accuracy)
and score v1's 2 test matches once, with v1's metric (third_party/birdseye_v1/action/train_final.evaluate_peaks).
usage: BFMD_DATA_ROOT=data/bfmd/v2 FEAT=shuttle python tools/bfmd_ensemble.py v2_clean_shuttle_sp2_ls0.1_self1acc_s{s} [SEEDS=0,1,2]"""
import json, os, sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bfmd_stroke as B  # noqa: E402
from hit_model_transformer import HitSpotterTransformer  # noqa: E402

tf = B.tf
if not torch.cuda.device_count():
    tf.DEVICE = torch.device("cpu")


class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x, pad_mask=None):
        hs, ss = zip(*[m(x, pad_mask=pad_mask) for m in self.models])
        p = torch.stack([torch.sigmoid(h) for h in hs]).mean(0).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p)), torch.log(torch.stack([torch.softmax(s, -1) for s in ss]).mean(0))


def main():
    pattern = sys.argv[1]
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
    d = json.load(open(f"{tf.BASE}/training_dataset_all_selfdetected.json"))
    B.set_pelvis(d)
    if "shuttle" in B.FEATS:
        B.add_shuttle(d)
    if "racket" in B.FEATS:
        B.add_racket(d)
        tf.LR_PERM = tf.LR_PERM + [len(tf.LR_PERM)]
    vocab = d["shot_type_vocab"]
    s2i = {s: i for i, s in enumerate(vocab)}
    _, test_tags, test_m = tf.load_split(d)
    mor = d["match_of_rally"]
    _, sel_m = B.split_matches(mor, set(mor.values()) - test_m)
    tags = sorted(d["rallies"])
    sel = tf.RallyDataset(d, [t for t in tags if mor[t] in sel_m], s2i)
    test = tf.RallyDataset(d, test_tags, s2i)
    in_dim = tf.make_features(*test.samples[0][:5]).shape[1]
    models = []
    for s in seeds:
        m = HitSpotterTransformer(in_dim=in_dim, d_model=192, n_heads=6, n_layers=4, n_shot_types=len(vocab))
        m.load_state_dict(torch.load(os.path.join(B.OUT, pattern.format(s=s) + ".pt"), map_location="cpu"))
        models.append(m.to(tf.DEVICE).eval())
    E = Ensemble(models).eval()
    t = max(B.THRESHOLDS, key=lambda t: sum(tf.evaluate_peaks(E, sel, t)[2:4]))
    p, r, f1, sacc, tp, fp, fn = tf.evaluate_peaks(E, test, t)
    tag = pattern.replace("_s{s}", "") + "_ens" + "".join(map(str, seeds))
    json.dump({"tag": tag, "members": [pattern.format(s=s) for s in seeds], "thresh": t, "P": p, "R": r, "F1": f1,
               "shot_acc": sacc, "tp": tp, "fp": fp, "fn": fn}, open(os.path.join(B.OUT, tag + ".json"), "w"), indent=1)
    print(f"RESULT {tag} [t={t}]: test P={p:.3f} R={r:.3f} F1={f1:.3f} shot_acc={sacc:.3f} (tp={tp} fp={fp} fn={fn})", flush=True)


if __name__ == "__main__":
    main()
