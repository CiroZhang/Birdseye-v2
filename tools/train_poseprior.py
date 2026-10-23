"""Train birdseye_v2.poseprior on the two-drone triangulated poses of DroneData's dev matches (outputs/pose3d).
Only poses with all 12 body joints triangulated are used; 10% of rallies are held out to report the
reconstruction error on unseen poses. Output: outputs/rallynet/poseprior.pt
usage: python tools/train_poseprior.py   (env EPOCHS=200)"""
import glob, os, sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.poseprior import PosePrior, canonical  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FLIP = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]  # body joints (COCO 5-16) left <-> right


def main():
    ps = sorted(glob.glob(os.path.join(ROOT, "outputs", "pose3d", "*", "*.npz")))
    rng = np.random.default_rng(0)
    held = set(rng.choice(len(ps), max(1, len(ps) // 10), replace=False))
    tr, te = [], []
    for i, p in enumerate(ps):
        J = np.load(p)["J"]
        J = J[np.isfinite(J[:, 5:]).all((1, 2))]
        (te if i in held else tr).append(J)
    tr, te = np.concatenate(tr), np.concatenate(te)
    print(len(tr), "train poses,", len(te), "held-out poses", flush=True)
    Xtr = canonical(torch.tensor(tr, device=DEV))
    Xte = canonical(torch.tensor(te, device=DEV))
    # drop triangulation outliers: limbs longer than any person's
    ok = lambda X: (X.norm(dim=-1).max(-1).values < 1.6)
    Xtr, Xte = Xtr[ok(Xtr)], Xte[ok(Xte)]
    m = PosePrior().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    E = int(os.environ.get("EPOCHS", "200"))
    for ep in range(E):
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(Xtr), 512):
            x = Xtr[perm[i:i + 512]]
            x = x * torch.empty(len(x), 1, 1, device=DEV).uniform_(0.92, 1.08)  # body size
            if torch.rand(()) < 0.5:  # left-right mirror (left-handed players): x -> -x, swap left/right joints
                x = x[:, FLIP] * torch.tensor([-1.0, 1.0, 1.0], device=DEV)
            loss = (m(x, noise=0.05) - x).norm(dim=-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == E - 1:
            with torch.no_grad():
                print(ep, "train %.3f m  held-out %.3f m  (mean per-joint reconstruction)" % (
                    (m(Xtr[:5000]) - Xtr[:5000]).norm(dim=-1).mean(), (m(Xte) - Xte).norm(dim=-1).mean()), flush=True)
    torch.save(m.state_dict(), os.path.join(ROOT, "outputs", "rallynet", "poseprior.pt"))


if __name__ == "__main__":
    main()
