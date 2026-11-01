import torch
import torch.nn as nn
import torch.nn.functional as F


class MeSH2d(nn.Module):
    """S memory slots per pixel. write: slot_b += u * w_b ; read: h = sum_b r_b * slot_b."""
    def __init__(self, C, S, T):
        super().__init__()
        self.S = S
        self.w_init, self.r_init = nn.Conv2d(C, S, 1), nn.Conv2d(C, S, 1)
        self.w = nn.ModuleList([nn.Conv2d(C, S, 1) for _ in range(T)])   # per-loop routers
        self.r = nn.ModuleList([nn.Conv2d(C, S, 1) for _ in range(T)])

    def init(self, x, v):
        B, C, H, W = x.shape
        M = torch.cat([x.unsqueeze(1), x.new_zeros(B, self.S - 1, C, H, W)], 1)   # slot 0 = anchor
        M = M + v.unsqueeze(1) * F.softmax(self.w_init(x), 1).unsqueeze(2)
        h = (M * F.softmax(self.r_init(x), 1).unsqueeze(2)).sum(1)
        return h, M

    def forward(self, u, h, M, t):
        M = M + u.unsqueeze(1) * F.softmax(self.w[t](h), 1).unsqueeze(2)           # write
        h = (M * F.softmax(self.r[t](h), 1).unsqueeze(2)).sum(1)                   # read
        return h, M


class Down2d(nn.Module):
    """g x g window -> 1 cell, softmax-weighted average (learned scorer)."""
    def __init__(self, C):
        super().__init__()
        self.score = nn.Conv2d(C, 1, 1)

    def forward(self, h, g):
        B, C, H, W = h.shape
        ph, pw = (-H) % g, (-W) % g
        if ph or pw:
            h = F.pad(h, (0, pw, 0, ph), mode="replicate")      # pad, don't drop
        H2, W2 = h.shape[2] // g, h.shape[3] // g
        a = self.score(h).reshape(B, 1, H2, g, W2, g).permute(0, 1, 2, 4, 3, 5).reshape(B, 1, H2, W2, g * g)
        x = h.reshape(B, C, H2, g, W2, g).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H2, W2, g * g)
        return (F.softmax(a, -1) * x).sum(-1)                    # (B, C, H2, W2)


class Up2d(nn.Module):
    """Each coarse cell -> its own g x g block; router beta decides the split inside the block."""
    def __init__(self, C, g):
        super().__init__()
        self.g = g
        self.router = nn.Conv2d(C, g * g, 1)

    def forward(self, z, H, W):
        g = self.g
        beta = F.pixel_shuffle(F.softmax(self.router(z), 1), g)           # (B,1,H2*g,W2*g)
        zup = z.repeat_interleave(g, 2).repeat_interleave(g, 3)           # (B,C,H2*g,W2*g)
        u = (g * beta) * zup               # gain g keeps the block's L2 energy (2D version of sqrt(g))
        return u[:, :, :H, :W]                                            # crop the padding


def make_core(C, k=1, gn_groups=8):
    layers = []
    for _ in range(k):
        layers += [nn.Conv2d(C, C, 3, padding=1, bias=False), nn.GroupNorm(gn_groups, C), nn.PReLU()]
    return nn.Sequential(*layers)


class LoopCNN(nn.Module):
    def __init__(self, C, sizes=(4, 2, 1), S=4, cores=None, k=1, gn_groups=8, tie_loops=True):
        super().__init__()
        self.sizes = list(sizes)
        T = len(self.sizes)
        if cores is None:                                   # নিজের core বানাও
            n = 1 if tie_loops else T
            cores = nn.ModuleList([make_core(C, k, gn_groups) for _ in range(n)])
        self.cores = cores                                  # ১টা (loop জুড়ে tied) বা T টা (প্রতি loop-এ একটা)
        self.down = nn.ModuleList([Down2d(C) if g > 1 else nn.Identity() for g in self.sizes])
        self.up = nn.ModuleList([Up2d(C, g) if g > 1 else nn.Identity() for g in self.sizes])
        self.mesh = MeSH2d(C, S, T)

    def core(self, t, x):
        return self.cores[t if len(self.cores) > 1 else 0](x)

    def forward(self, p):
        H, W = p.shape[2:]
        h, M = self.mesh.init(p, p)
        for t, g in enumerate(self.sizes):
            if g == 1:
                u = self.core(t, h)
            else:
                u = self.up[t](self.core(t, self.down[t](h, g)), H, W)
            h, M = self.mesh(u, h, M, t)
        return h


class LoopCNNFace(nn.Module):
    def __init__(self, channels=64, sizes=((4, 2, 1),) * 3, num_slots=4, k=1,
                 tie_levels=True, tie_loops=True, gn_groups=8):
        super().__init__()
        T = len(sizes[0])
        shared = None
        if tie_levels:                                      # সব level একই core(গুলো) ব্যবহার করে
            assert all(len(s) == T for s in sizes)
            n = 1 if tie_loops else T
            shared = nn.ModuleList([make_core(channels, k, gn_groups) for _ in range(n)])
        self.levels = nn.ModuleList(
            [LoopCNN(channels, s, num_slots, shared, k, gn_groups, tie_loops) for s in sizes])

    def forward(self, feats, L=None):
        return [lvl(p) for lvl, p in zip(self.levels, feats)]