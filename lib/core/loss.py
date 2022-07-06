# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from hhrnet.lib.core.tag_loss import BaseTagLoss, KLDTagLoss, JensenShannonTagLoss, WassersteinTagLoss, MMDTagLoss

logger = logging.getLogger(__name__)


def make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp.cuda()
    return inp

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask=None):
        assert pred.size() == gt.size()
        loss = 0
        if mask is not None:
            loss = ((pred - gt)**2) * mask[:, None, :, :].expand_as(pred)
        else:
            loss = ((pred - gt)**2)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        # loss = loss.mean(dim=3).mean(dim=2).sum(dim=1)
        return loss

class HighDimAELoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loss_type = cfg.LOSS.AE_LOSS_TYPE
        self.joint_distance_ae = cfg.LOSS.JOINT_DISTANCE_AE
        self.ae_prior = cfg.LOSS.AE_PRIOR
        self.ae_dim = cfg.MODEL.NUM_JOINTS

        if self.ae_prior != '': # assume truncated gaussian in range [0,1]
            ae_bins = torch.arange(self.ae_dim)/self.ae_dim
            sigma = cfg.LOSS.AE_PRIOR_SIGMA
            self.gt_prior = torch.exp(-0.5 * ae_bins**2 / (sigma ** 2)) # 17 size
            self.ae_prior_loss = nn.KLDivLoss(reduction='none')

        if self.loss_type == 'exp': # base setting
            self.loss = BaseTagLoss()
        elif self.loss_type == 'kld':
            self.loss = KLDTagLoss()
        elif self.loss_type == 'jensen_shannon':
            self.loss = JensenShannonTagLoss()
        elif self.loss_type == 'wasserstein':
            self.loss = WassersteinTagLoss()
        elif self.loss_type == 'mmd':
            self.loss = MMDTagLoss()

    # pred_tag: [17, 128, 128]
    # joints: [max_people, 17, 2]. [:,:,0]:joint position in 1d. [:,:,1]:visibility
    # heatmap: use for gaussian masking
    def multiTagLoss(self, pred_tag, joints, heatmap=None):
        tags = []
        pull = 0
        push = 0
        output_res = pred_tag.shape[1]

        # pull
        for joints_per_person in joints:
            valid_joints = joints_per_person[joints_per_person[:,1]>0]
            if len(valid_joints)==0:
                continue

            valid_joints_offset = valid_joints[:,0] % (output_res ** 2) # remove joint dim idx
            # assume x as col idx, y as row idx
            valid_joints_2d = np.stack((valid_joints_offset%output_res, valid_joints_offset//output_res),axis=1)
            valid_joints_2d = valid_joints_2d.astype(np.int32)
            valid_joints_2d = np.clip(valid_joints_2d, 0, pred_tag.shape[1]-1)

            try:
                tags_per_person = torch.stack([pred_tag[:,joint[0], joint[1]] for joint in valid_joints_2d]) # [n_valid_kpt, ae_dim]
            except IndexError  as e:
                print("OOB during tag loss computation")
                import ipdb; ipdb.set_trace()

            tags.append(torch.mean(tags_per_person, dim=0))

            n_valid_kpt = tags_per_person.shape[1]
            if n_valid_kpt<=1: # we don't have to calculate pull loss for single joint
                continue
            tag_mat = tags_per_person.T.unsqueeze(0).repeat(n_valid_kpt,1,1) # [n_kpt, n_kpt, ae_dim]
            pull_per_person = self.loss(tag_mat, torch.transpose(tag_mat,0,1))
            pull = pull + pull_per_person

        # average
        n_tags = len(tags)

        if n_tags == 0:
            return make_input(torch.zeros(1).float()), \
                make_input(torch.zeros(1).float())
        elif n_tags == 1:
            return make_input(torch.zeros(1).float()), \
                pull/(n_tags)

        pull = pull / n_tags

        # push
        tags = torch.stack(tags) # [n_tags, ae_dim]
        tags_mat = tags.unsqueeze(0).repeat(n_tags,1,1) # [n_tags, n_tags, ae_dim]
        push = self.loss(tags_mat, torch.transpose(tags_mat,0,1), exp=True, push=True)
        """
        if self.loss_type == 'exp':
            push = self.loss(tags_mat, torch.transpose(tags_mat,0,1), exp=True)
        else:
            push = self.loss(tags_mat, torch.transpose(tags_mat,0,1))
        """
        return push, pull

    # tags: [b, 17, 128, 128, 1]
    # joints: [b, 30(max_people), 17, 3]
    def forward(self, tags, joints, heatmap=None):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            if heatmap is not None:
                push, pull = self.multiTagLoss(tags[i], joints[i], heatmap[i])
            else:
                push, pull = self.multiTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)

        if hasattr(self,'ae_prior_loss') and self.ae_prior_loss is not None:
            gt_batch_prior = self.gt_prior.repeat(tags[0].shape, 1, *tags[2:].shape)
            pred = F.log_softmax(tags, dim=1) # map pred to log space. softmax over ae_dim
            target = F.softmax(gt_batch_prior, dim=1) # softmax over ae_dim
            prior_loss = self.ae_prior_loss(pred, target)
            prior_loss = torch.sum(prior_loss, dim=1)# reduce tag dim
            prior_loss = torch.sum(prior_loss, dim=(1,2)) # reduce except batch dim
            prior_loss = torch.clamp(prior_loss, min=0,max=tags.shape[1])
            return torch.stack(pushes), torch.stack(pulls), prior_loss
        else:
            return torch.stack(pushes), torch.stack(pulls)

class AELoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    # pred_tag: [flattened heatmap size, 1]
    # joints: [batch_size, 17, 2]
    # heatmaps: not used
    def singleTagLoss(self, pred_tag, joints, heatmaps=None):
        """
        associative embedding loss for one image
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        num_tags = len(tags) #

        if num_tags == 0:
            return make_input(torch.zeros(1).float()), \
                make_input(torch.zeros(1).float())
        elif num_tags == 1:
            return make_input(torch.zeros(1).float()), \
                pull/(num_tags)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp': # max: 1(diagonal entries)
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unkown ae loss type')

        if torch.sum(torch.isnan(diff)):
            print('nan detected during single tag AELoss computation')
            import ipdb; ipdb.set_trace()

        return push/((num_tags - 1) * num_tags) * 0.5, \
            pull/(num_tags)
    # tags: [b, 17 * 128 * 128, 1]
    # joints: [b, 30(max_people), 17, 2]
    def forward(self, tags, joints, heatmaps=None):
        """
        accumulate the tag loss for each image in the batch
        """
        batch_size = tags.size()[0]
        tags = tags.contiguous().view(batch_size, -1, 1)

        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)

        return torch.stack(pushes), torch.stack(pulls)


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class LossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.heatmaps_loss = None
        self.ae_loss = None
        self.heatmaps_loss_factor = 1.0
        self.push_loss_factor = 1.0
        self.pull_loss_factor = 1.0

        if cfg.LOSS.WITH_HEATMAPS_LOSS:
            self.heatmaps_loss = HeatmapLoss()
            self.heatmaps_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR
        if cfg.LOSS.WITH_AE_LOSS:
            self.ae_loss = AELoss(cfg.LOSS.AE_LOSS_TYPE)
            self.push_loss_factor = cfg.LOSS.PUSH_LOSS_FACTOR
            self.pull_loss_factor = cfg.LOSS.PULL_LOSS_FACTOR

        if not self.heatmaps_loss and not self.ae_loss:
            logger.error('At least enable one loss!')

    def forward(self, outputs, heatmaps, masks, joints):
        # TODO(bowen): outputs and heatmaps can be lists of same length
        heatmaps_pred = outputs[:, :self.num_joints]
        tags_pred = outputs[:, self.num_joints:]

        heatmaps_loss = None
        push_loss = None
        pull_loss = None

        if self.heatmaps_loss is not None:
            heatmaps_loss = self.heatmaps_loss(heatmaps_pred, heatmaps, masks)
            heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor

        if self.ae_loss is not None:
            batch_size = tags_pred.size()[0]
            tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

            push_loss, pull_loss = self.ae_loss(tags_pred, joints)
            push_loss = push_loss * self.push_loss_factor
            pull_loss = pull_loss * self.pull_loss_factor

        return [heatmaps_loss], [push_loss], [pull_loss]


class MultiLossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # init check
        self._init_check(cfg)

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_stages = cfg.LOSS.NUM_STAGES

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in cfg.LOSS.WITH_HEATMAPS_LOSS
                ]
            )
        self.heatmaps_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR

        if cfg.LOSS.HIGH_DIM_AE:
            self.ae_prior = True if cfg.LOSS.AE_PRIOR!='' else False
            self.ae_loss = \
                nn.ModuleList(
                    [
                        HighDimAELoss(cfg) if with_ae_loss else None
                        for with_ae_loss in cfg.LOSS.WITH_AE_LOSS
                    ]
                )
        else:
            self.ae_loss = \
                nn.ModuleList(
                    [
                        AELoss(cfg.LOSS.AE_LOSS_TYPE) if with_ae_loss else None
                        for with_ae_loss in cfg.LOSS.WITH_AE_LOSS
                    ]
                )
        self.push_loss_factor = cfg.LOSS.PUSH_LOSS_FACTOR
        self.pull_loss_factor = cfg.LOSS.PULL_LOSS_FACTOR
        self.prior_loss_factor = cfg.LOSS.PRIOR_LOSS_FACTOR

    def forward(self, outputs, heatmaps, masks, joints):
        # forward check
        self._forward_check(outputs, heatmaps, masks, joints)

        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        prior_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints

                heatmaps_loss = self.heatmaps_loss[idx](
                    heatmaps_pred, heatmaps[idx], masks[idx]
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)

            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]

                if hasattr(self, 'ae_prior') and self.ae_prior:
                    push_loss, pull_loss, prior_loss = self.ae_loss[idx](
                        tags_pred, joints[idx], heatmaps[idx]
                    )
                    prior_loss = prior_loss * self.prior_loss_factor[idx]
                    prior_losses.append(prior_loss)
                else:
                    prior_losses.append(None)
                    push_loss, pull_loss = self.ae_loss[idx](
                        tags_pred, joints[idx], heatmaps[idx]
                    )
                    push_loss = push_loss * self.push_loss_factor[idx]
                    pull_loss = pull_loss * self.pull_loss_factor[idx]

                    push_losses.append(push_loss)
                    pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)
                prior_losses.append(None)

        return heatmaps_losses, push_losses, pull_losses, prior_losses

    def _init_check(self, cfg):
        assert isinstance(cfg.LOSS.WITH_HEATMAPS_LOSS, (list, tuple)), \
            'LOSS.WITH_HEATMAPS_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.HEATMAPS_LOSS_FACTOR, (list, tuple)), \
            'LOSS.HEATMAPS_LOSS_FACTOR should be a list or tuple'
        assert isinstance(cfg.LOSS.WITH_AE_LOSS, (list, tuple)), \
            'LOSS.WITH_AE_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)), \
            'LOSS.PUSH_LOSS_FACTOR should be a list or tuple'
        assert isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)), \
            'LOSS.PUSH_LOSS_FACTOR should be a list or tuple'
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == len(cfg.LOSS.HEATMAPS_LOSS_FACTOR), \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.HEATMAPS_LOSS_FACTOR should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), len(cfg.LOSS.HEATMAPS_LOSS_FACTOR))
        assert len(cfg.LOSS.WITH_AE_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_AE_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_AE_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_AE_LOSS) == len(cfg.LOSS.PUSH_LOSS_FACTOR), \
            'LOSS.WITH_AE_LOSS and LOSS.PUSH_LOSS_FACTOR should have same length, got {} vs {}.'. \
                format(len(cfg.LOSS.WITH_AE_LOSS), len(cfg.LOSS.PUSH_LOSS_FACTOR))
        assert len(cfg.LOSS.WITH_AE_LOSS) == len(cfg.LOSS.PULL_LOSS_FACTOR), \
            'LOSS.WITH_AE_LOSS and LOSS.PULL_LOSS_FACTOR should have same length, got {} vs {}.'. \
                format(len(cfg.LOSS.WITH_AE_LOSS), len(cfg.LOSS.PULL_LOSS_FACTOR))

    def _forward_check(self, outputs, heatmaps, masks, joints):
        assert isinstance(outputs, list), \
            'outputs should be a list, got {} instead.'.format(type(outputs))
        assert isinstance(heatmaps, list), \
            'heatmaps should be a list, got {} instead.'.format(type(heatmaps))
        assert isinstance(masks, list), \
            'masks should be a list, got {} instead.'.format(type(masks))
        assert isinstance(joints, list), \
            'joints should be a list, got {} instead.'.format(type(joints))
        assert len(outputs) == self.num_stages, \
            'len(outputs) and num_stages should been same, got {} vs {}.'.format(len(outputs), self.num_stages)
        assert len(outputs) == len(heatmaps), \
            'outputs and heatmaps should have same length, got {} vs {}.'.format(len(outputs), len(heatmaps))
        assert len(outputs) == len(masks), \
            'outputs and masks should have same length, got {} vs {}.'.format(len(outputs), len(masks))
        assert len(outputs) == len(joints), \
            'outputs and joints should have same length, got {} vs {}.'.format(len(outputs), len(joints))
        assert len(outputs) == len(self.heatmaps_loss), \
            'outputs and heatmaps_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.heatmaps_loss))
        assert len(outputs) == len(self.ae_loss), \
            'outputs and ae_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.ae_loss))


def test_ae_loss():
    import numpy as np
    t = torch.tensor(
        np.arange(0, 32).reshape(1, 2, 4, 4).astype(float)*0.1,
        requires_grad=True
    )
    t.register_hook(lambda x: print('t', x))

    ae_loss = AELoss(loss_type='exp')

    joints = np.zeros((2, 2, 2))
    joints[0, 0] = (3, 1)
    joints[1, 0] = (10, 1)
    joints[0, 1] = (22, 1)
    joints[1, 1] = (30, 1)
    joints = torch.LongTensor(joints)
    joints = joints.view(1, 2, 2, 2)

    t = t.contiguous().view(1, -1, 1)
    l = ae_loss(t, joints)

    print(l)

def test_high_dim_ae_loss():
    import numpy as np
    # tag: [b, n_kpt, w, h, 1]
    ae_dim = 17
    w, h = 8,8
    t = torch.tensor(
        np.arange(0, ae_dim*w*h).reshape(1, 17, w, h).astype(float)/(ae_dim*w*h),
        requires_grad=True
    )
    t.register_hook(lambda x: print('t', x))

    #ae_loss = HighDimAELoss(loss_type='exp')
    #ae_loss = HighDimAELoss(loss_type='kld')
    ae_loss = HighDimAELoss(loss_type='wasserstein')

    joints = np.zeros((2, 17, 3))
    for i in range(2):
        for j in range(17):
            joints[i,j] = (i*3,1,1)

    joints = torch.LongTensor(joints)
    joints = joints.view(1, 2, 17, 3) # [b, max_people, n_kpt, (x,y,sigma)]

    l = ae_loss(t, joints)

    print(l)

def test_prior_loss():
    t = torch.arange(10, dtype=float)/10
    t.requires_grad=True
    t.register_hook(lambda x: print('t', x))

    prior_loss = nn.KLDivLoss(reduction='none')

    pred = torch.randperm(10)/10

    l = prior_loss(pred, t)

    import ipdb; ipdb.set_trace()

    print(l)

if __name__ == '__main__':
    #test_ae_loss()
    #test_high_dim_ae_loss()
    test_prior_loss()
