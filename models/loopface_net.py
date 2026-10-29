#
# LoopFace Author:
# Copyright (c) 2026 Yasin Arafat Licensed under the MIT License, see LICENSE
#
""" 
We can't use only the haar transformation alone. Cause, after sate-1 for MobileNetV1 output channel, 64. 
and the image size does not match. Haar transformation(B, C, H, W) -> (B, 4C, H/2, W/2) do this. So, 
we need to match the size. If we wanna add. Another problem, is haar doesn't learn anything this the problem.
From nn we get output that's is normalized but the vlaue from haar is not. So, we'll add somthing, that
will decide which part of the image we needed. 

"""

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from loopface.loop import LoopFace
from collections import OrderedDict
from loopface.haar import HaarTransform
from  models.retinaface import RetinaFace
from loopface.dilated_loop import DilatedLoopFace

class HaarBranch(nn.Module):
    """ 
    widths = channel dimention each loop we need 64.
    """
    def __init__(self,in_channels=3,widths=(16,32,64)):
        super().__init__()
        layers, c = [], in_channels
        for i,w in enumerate(widths):
            #             not-learable          learable                    norm
            layers += [HaarTransform(c), nn.Conv2d(4*c,w,1,bias=False),nn.BatchNorm2d(w)]
            # add a activation funtion at the end layers:
            if i<len(widths) -1:
                layers.append(nn.LeakyReLU(0.1,inplace=True))
            # channel is need for the next step
            c = w 
        self.net = nn.Sequential(*layers)
        # batch norm: weight = 0 
        # output = γ × (normalized input) + β
        # Goyal et al. (2017, “Accurate, Large Minibatch SGD”)-> ResNet block- BN- γ=0
        # after traning finish, if γ is big then haar info worked else not.
        nn.init.zeros_(self.net[-1].weight)

    def forward(self,x):
        return self.net(x)



class HaarBody(nn.Module):
    """ Takes the pieces out of `backbone` (an IntermediateLayerGetter or the raw model),
    keeps their names, so the pretrained weights load as before.
    Returns {1: C3, 2: C4, 3: C5} exactly like IntermediateLayerGetter.
    """
    def __init__(self,backbone,name):
        super().__init__()
        self.name = name
        if name == 'mobilenet0.25':
            self.stage1, self.stage2, self.stage3 = backbone.stage1, backbone.stage2, backbone.stage3
            self.haar = HaarBranch(3,widths=(16,32,64))
        elif name == "Resnet50":
            self.conv1, self.bn1 = backbone.conv1, backbone.bn1
            self.relu, self.maxpool = backbone.relu, backbone.maxpool
            self.layer1, self.layer2 = backbone.layer1, backbone.layer2
            self.layer3, self.layer4 = backbone.layer3, backbone.layer4
            self.haar = HaarBranch(3, widths=(16, 64)) 
        else:
            raise ValueError(f"HaarBody: unknown backbone {name}")

    def forward(self,x):
        if self.name == 'mobilenet0.25':
            c3 = self.stage1(x) + self.haar(x)    # [B, 64, 80, 80]
            c4 = self.stage2(c3)                  # [B, 128, 40, 40]
            c5 = self.stage3(c4)                  # [B, 256, 20, 20]
        else:
            x0 = self.maxpool(self.relu(self.bn1(self.conv1(x)))) + self.haar(x)
            c3 = self.layer2(self.layer1(x0))
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
        return OrderedDict([(1, c3), (2, c4), (3, c5)]) 


class LoopFaceNet(RetinaFace):
    def __init__(self, cfg=None, phase='train'):
        super().__init__(cfg, phase)

        if cfg.get('haar',False):
            self.body = HaarBody(self.body,cfg['name'])


        if cfg.get('loop'):
            if cfg.get('loop_dilations'):
                self.loop = DilatedLoopFace(cfg['out_channel'],
                                            train_L=tuple(cfg.get('loop_train_L', (1, 6))),
                                            test_L=cfg.get('loop_test_L', 4),
                                            dilations=tuple(cfg['loop_dilations']))
            else:
                self.loop = LoopFace(cfg['out_channel'],
                                     train_L=tuple(cfg.get('loop_train_L', (1, 6))),
                                     test_L=cfg.get('loop_test_L', 4))
        else:
            self.loop = None 

        self.use_ssh = cfg.get('ssh', True)
        if not self.use_ssh:            
            self.ssh1 = self.ssh2 = self.ssh3 = None

    def forward(self, inputs,L=None):
        out = self.body(inputs)
        fpn = self.fpn(out)

        if self.use_ssh:
            features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        else:
            features = list(fpn)

        if self.loop is not None: 
            features = self.loop(features,L)


        bbox = torch.cat([self.BboxHead[i](f) for i, f in enumerate(features)], dim=1)
        cls = torch.cat([self.ClassHead[i](f) for i, f in enumerate(features)], dim=1)
        ldm = torch.cat([self.LandmarkHead[i](f) for i, f in enumerate(features)], dim=1)

        if self.phase == 'train':
            return bbox, cls, ldm
        return bbox, F.softmax(cls, dim=-1), ldm

