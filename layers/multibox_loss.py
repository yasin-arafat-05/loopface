#
# Source: https://github.com/biubug6/Pytorch_Retinaface (layers/multibox_loss.py)
# Copyright (c) 2019 Pytorch_Retinaface authors. MIT License, see LICENSE.MIT
# Modified by loopface authors. Copyright (c) 2026 Yasin Arafat. MIT License, see LICENSE
#
import torch
import torch.nn as nn
from data import cfg_mnet
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp


GPU = cfg_mnet['gpu_train']


""" 
What's our models say and what's our match() say, we compare this and calcuate loss. 
"""
class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        - utils->box_utils.py->match() function does (1) and (2). (3)-here.
        1) Produce Confidence Target(conf_t from match()) Indices by matching  ground truth boxes
           with (default is SSD paper, we said it anchor or priorboxes) 'priorboxes' that have jaccard(IoU)
           index > threshold parameter(default threshold: 0.5).
        2) Produce localization target(loc_t from match()) by 'encoding' variance into offsets(encode() in box_utils.py) of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α(importance of box loss) which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes(positive anchor)
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    And RetinFace add loss_landm, that's not present in SSD Formula.
    
    Pipeline:
    Image ──→ model ──→ loc_data, conf_data, landm_data ──┐
                                                          ├──→ loss ──→ backward
    GT + anchor ──→ match() ──→ loc_t, conf_t, landm_t   ─┘
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # model prediction: loc_data(bbox-shift), conf_data(confidence,class)
        # landm_data: landmark-shift. why shift not the actual, bbox?
        # During training, the model learns that the correct shift for this anchor is encode(face box, anchor). 
        # So at inference, when it outputs a shift, running decode on that shift gives back the face’s box.
        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # ====== step1: make targets without the nn with the -> match() =====
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            # match() return nothing, just update things.
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        # ====== step2: Finding the landmark loss =====
        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        # from conft_t take only (1's), there might be also -1 in our conft_t
        # the face those haven't landmark, there is no need calculate loss for them.
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        # landmark from network:
        landm_p = landm_data[pos_idx1].view(-1, 10)
        # langmark from match() funtion.(return the shift from actual landmarks)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        # ======= step:3 Find the box loss ===============
        pos = conf_t != zeros
        # make conf_t -1 value into 1.
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4) # from match()
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # ======= step:4 class loss and negative mining =========
        # Compute max conf across batch for hard negative mining 
        # why negative -> if model predict always positive then there 
        # might be change model predict backgournd as positive. As we 
        # are giving negative model can easily differenctiate among them. And: 
        # NMS can't fix this. 
        batch_conf = conf_data.view(-1, self.num_classes)
        
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        # conft_t =1, positive, 0 = negative.
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        # cross entropy loss:
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        # box_loss and landmark = negative not need, class_loss = negative needed
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm