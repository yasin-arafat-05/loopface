import torch
from data.config import cfg_mnet
from models.retinaface import RetinaFace
from models.loopface_net import LoopFaceNet
from data.config import cfg_mnet_loop_nossh, cfg_mnet_dloop_nossh

cfg = dict(cfg_mnet, pretrain=False)
sd = torch.load('./weights/mobilenet0.25_Final.pth', map_location='cpu', weights_only=False)
sd = {k.replace('module.', ''): v for k, v in sd.items()}

a = RetinaFace(cfg=cfg, phase='test').eval();  a.load_state_dict(sd)
b = LoopFaceNet(cfg=cfg, phase='test').eval(); b.load_state_dict(sd)



def test_dilated_loop():
    """Shapes for L = 1, 4, 16; same parameter count as the plain loop; identical output at L = 1
    (dilation 1 == plain 3x3) and different output at L = 2 (dilation 2 kicks in)."""
    torch.manual_seed(0)
    plain = LoopFaceNet(cfg=dict(cfg_mnet_loop_nossh, pretrain=False), phase='test').eval()
    dil = LoopFaceNet(cfg=dict(cfg_mnet_dloop_nossh, pretrain=False), phase='test').eval()
    dil.load_state_dict(plain.state_dict(), strict=True)            # same weights in both
    n = lambda m: sum(p.numel() for p in m.parameters())
    x = torch.randn(1, 3, 640, 640)
    ok = n(plain) == n(dil)
    with torch.no_grad():
        for L in (1, 4, 16):
            loc, conf, ldm = dil(x, L=L)
            ok &= tuple(loc.shape) == (1, 16800, 4) and tuple(conf.shape) == (1, 16800, 2) \
                  and tuple(ldm.shape) == (1, 16800, 10)
        ok &= torch.allclose(plain(x, L=1)[0], dil(x, L=1)[0], atol=1e-6)
        ok &= not torch.allclose(plain(x, L=2)[0], dil(x, L=2)[0], atol=1e-6)
    return bool(ok)

x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    print([torch.allclose(p, q, atol=1e-6) for p, q in zip(a(x), b(x))])   # [True, True, True]
    print(test_dilated_loop())

