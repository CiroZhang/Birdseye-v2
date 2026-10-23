"""A learned prior over badminton body poses from our own two-drone triangulation (tools/triangulate_poses.py,
DroneData dev matches only): an autoencoder over body poses in the body's own frame (pelvis at the origin,
hips along x, z up). A pose it cannot reconstruct is unlike anything the players were seen doing, so its
reconstruction error is a plausibility penalty for RallyNet (rallynet.losses, frozen prior).

Only the 12 body joints (shoulders to ankles, COCO 5-16): seen from behind, the face is rarely triangulated.
canonical(J): (...,17,3) -> (...,12,3); PosePrior(Xc) -> reconstruction in the same frame."""
import torch
import torch.nn as nn


def canonical(J):
    """Pelvis-centred, rotated about z so the hip line runs along +x (left hip -> right hip)."""
    pel = 0.5 * (J[..., 11, :] + J[..., 12, :])
    X = J - pel[..., None, :]
    h = X[..., 12, :2] - X[..., 11, :2]
    short = h.norm(dim=-1, keepdim=True) < 1e-4  # empty slot / collapsed hips: atan2(0, 0) has a NaN gradient
    h = torch.where(short, torch.tensor([1.0, 0.0], device=h.device, dtype=h.dtype), h)
    a = torch.atan2(h[..., 1], h[..., 0])
    c, s = torch.cos(-a), torch.sin(-a)
    x = c[..., None] * X[..., 0] - s[..., None] * X[..., 1]
    y = s[..., None] * X[..., 0] + c[..., None] * X[..., 1]
    return torch.stack([x, y, X[..., 2]], -1)[..., 5:, :]


class PosePrior(nn.Module):
    def __init__(self, z=16, h=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(36, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 36))

    def forward(self, Xc, noise=0.0):
        z = self.enc(Xc.flatten(-2))
        if noise:
            z = z + noise * torch.randn_like(z)
        return self.dec(z).reshape(Xc.shape)

    def penalty(self, J):
        """Mean per-joint distance (m) between a pose (court frame) and its reconstruction, per pose."""
        Xc = canonical(J)
        return (self(Xc) - Xc).norm(dim=-1).mean(-1)
