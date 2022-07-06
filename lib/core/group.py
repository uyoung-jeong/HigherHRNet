# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Some code is from https://github.com/princeton-vl/pose-ae-train/blob/454d4ba113bbb9775d4dc259ef5e6c07c2ceed54/utils/group.py
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from munkres import Munkres
import numpy as np
import torch


def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp

def match_by_high_dim_tag(inp, params, method='base'):
    assert isinstance(params, Params), 'params should be class Params()'
    # tag_k.shpae: [17, 30, 1] wo flip. [17, 30, 2] with flip
    # tag_k.shape: [17, 30, 17, 2]
    # loc_k.shape: [17, 30, 2]
    # val_k.shape: [17, 30]
    tag_k, loc_k, val_k = inp

    # prepare tag
    if params.normalize == 'naive': # simplest normalization
        #tag_k = tag_k - np.max(tag_k) # prevent overflow during softmax. cannot prevent underflow
        #tag_k = np.exp(tag_k) / np.expand_dims(np.sum(np.exp(tag_k), axis=2),axis=2) # softmax over tag_dim
        tag_k = tag_k - np.min(tag_k) # prevent negative value
        tag_k = tag_k / np.expand_dims(np.sum(np.exp(tag_k), axis=2),axis=2) # simple normalization

    if np.isnan(tag_k).any() or np.isinf(tag_k).any():
        print('nan or inf detected')
        import ipdb; ipdb.set_trace()
        print()

    tag_k = tag_k.reshape((*tag_k.shape[:-2],-1)) # [17, 30, 17 x (flip)]
    #tag_swap = np.swapaxes(tag_k,0,1).reshape((params.max_num_people,params.num_joints)) # [30, 17]
    #tag_swap = np.swapaxes(tag_k,0,1).reshape((params.max_num_people,-1)) # [30, 17] wo flip. [30, 34] with flip

    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2])) # [17, 4]

    if method != 'base':
        tag_swap_repeat = np.stack(tag_swap for _ in range(params.num_joints)) # [17, 30, 17]

        joints = np.concatenate((loc_k, val_k[:, :, None], tag_swap_repeat), 2) # [17,30, 3 + 17]
        mask = joints[:,:,2] > params.detection_threshold # (17, 30)

        joints = [joint[m] for joint,m in zip(joints, mask)] # 17 x [n_above_thres, 20]
        tags = [tag[m] for tag,m in zip(tag_swap_repeat, mask)] # 17 x [n_above_thres, 17]

        joint_dict = {} # {tag:joint}
        tag_dict = {}

        import ipdb; ipdb.set_trace()

        tags_mat = np.stack(tags)
        diff = tags_mat-tags_mat.T
        diff_normed = np.linalg.norm(diff, ord=2, axis=2)

        num_added = diff.shape[0]
        num_grouped = diff.shape[1]

        if num_added > num_grouped:
            diff_normed = np.concatenate(
                (
                    diff_normed,
                    np.zeros((num_added, num_added-num_grouped))+1e10
                ),
                axis=1
            )

        pairs = py_max_match(diff_normed) # [n_grouped, 2]

        # filtering pairs

        ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32) # [n_detect, 17, 4]
        return ans

    ### original
    joint_dict = {} # {tag:joint}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]

        #tags = tag_k[idx]
        tags = tag_k[idx] # [30, 17(tag_dim) x 2(flip)]

        joints = np.concatenate((loc_k[idx], val_k[idx, :, None], tags), 1) # [30, 3 + 17]
        mask = joints[:, 2] > params.detection_threshold # (30, )

        tags = tags[mask] # [n_above_thres,17]
        joints = joints[mask] # [n_above_thres, 3+17]

        if joints.shape[0] == 0: # no above thres
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            if params.ignore_too_much \
               and len(grouped_keys) == params.max_num_people:
                continue

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :] #[n_above_thres,n_grouped_tags,17]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2) #[n_above_thres,n_grouped_tags]

            diff_saved = np.copy(diff_normed)

            if params.use_detection_val: # True
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3] #

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (
                        diff_normed,
                        np.zeros((num_added, num_added-num_grouped))+1e10
                    ),
                    axis=1
                )

            pairs = py_max_match(diff_normed) # [n_grouped, 2]

            for row, col in pairs:
                if (
                    row < num_added
                    and col < num_grouped
                    and diff_saved[row][col] < params.tag_threshold
                ):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32) # [n_detect, 17, 4]
    return ans

def match_by_tag(inp, params):
    assert isinstance(params, Params), 'params should be class Params()'
    # tag_k.shpae: [17, 30, 1] wo flip. [17, 30, 2] with flip
    # loc_k.shape: [17, 30, 2]
    # val_k.shape: [17, 30]
    tag_k, loc_k, val_k = inp
    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2])) # [17, 4]

    joint_dict = {} # {tag:joint}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]

        tags = tag_k[idx] # [30,1]
        joints = np.concatenate(
            (loc_k[idx], val_k[idx, :, None], tags), 1
        ) # [30,4]
        mask = joints[:, 2] > params.detection_threshold # (30, )

        tags = tags[mask] # [n_above_thres,1] wo flip. [n_above_thres, 2] with flip
        joints = joints[mask] # [n_above_thres, 4]

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0] # unflipped one?
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            if params.ignore_too_much \
               and len(grouped_keys) == params.max_num_people:
                continue

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :] #[n_above_thres,n_grouped_tags,1]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2) #[n_above_thres,n_grouped_tags]
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val: # True
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3] #

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (
                        diff_normed,
                        np.zeros((num_added, num_added-num_grouped))+1e10
                    ),
                    axis=1
                )

            pairs = py_max_match(diff_normed) # [n_grouped, 2]

            for row, col in pairs:
                if (
                    row < num_added
                    and col < num_grouped
                    and diff_saved[row][col] < params.tag_threshold
                ):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32) # [n_detect, 17, 4] wo flip. [n_detect, 17, 5] with flip
    return ans


class Params(object):
    def __init__(self, cfg):
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.max_num_people = cfg.DATASET.MAX_NUM_PEOPLE

        self.detection_threshold = cfg.TEST.DETECTION_THRESHOLD
        self.tag_threshold = cfg.TEST.TAG_THRESHOLD
        self.use_detection_val = cfg.TEST.USE_DETECTION_VAL
        self.ignore_too_much = cfg.TEST.IGNORE_TOO_MUCH

        if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
            self.num_joints -= 1

        if cfg.DATASET.WITH_CENTER and not cfg.TEST.IGNORE_CENTER:
            self.joint_order = [
                i-1 for i in [18, 1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
        else:
            self.joint_order = [
                i-1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]

        self.high_dim_ae = cfg.LOSS.HIGH_DIM_AE
        self.normalize = cfg.TEST.NORMALIZE


class HeatmapParser(object):
    def __init__(self, cfg):
        self.params = Params(cfg)
        self.tag_per_joint = cfg.MODEL.TAG_PER_JOINT
        self.pool = torch.nn.MaxPool2d(
            cfg.TEST.NMS_KERNEL, 1, cfg.TEST.NMS_PADDING
        )

    def nms(self, det):
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def match(self, tag_k, loc_k, val_k):
        if self.params.high_dim_ae:
            match = lambda x: match_by_high_dim_tag(x, self.params)
        else:
            match = lambda x: match_by_tag(x, self.params)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def top_k(self, det, tag):
        # det = torch.Tensor(det, requires_grad=False)
        # tag = torch.Tensor(tag, requires_grad=False)

        det = self.nms(det)
        num_images = det.size(0)
        num_joints = det.size(1)
        h = det.size(2)
        w = det.size(3)
        det = det.view(num_images, num_joints, -1) # [1, 17, 393216]
        # val_k: [1, 17, 30]
        # ind: [1, 17, 30]
        val_k, ind = det.topk(self.params.max_num_people, dim=2)

        tag = tag.view(tag.size(0), tag.size(1), w*h, -1) # [1, 17, 393216, 1]
        if not self.tag_per_joint:
            tag = tag.expand(-1, self.params.num_joints, -1, -1)

        if self.params.high_dim_ae:
            tag_list = []
            for b_i in range(tag.size(0)): # batch dimension
                flip_wise_tag = []
                for i in range(tag.size(3)): # flip dimension
                    kpt_wise_tag = []
                    for k_i in range(ind.shape[1]): # keypoint dimension
                        kpt_wise_tag.append(tag[b_i,:,ind[b_i,k_i],i]) # [17, 30]
                    flip_wise_tag.append(torch.stack(kpt_wise_tag,dim=2)) # [17, 30, 17]
                tag_list.append(torch.stack(flip_wise_tag,dim=3)) # [17, 30, 17, 2]
            tag_k = torch.stack(tag_list, dim=0) # [1, 17, 30, 17, 2]
        else:
            tag_k = torch.stack(
                [
                    torch.gather(tag[:, :, :, i], 2, ind) # [1, 17, 30]
                    for i in range(tag.size(3))
                ],
                dim=3
            )

        x = ind % w # [1, 17, 30]
        y = (ind / w).long() # [1, 17, 30]

        ind_k = torch.stack((x, y), dim=3) # [1, 17, 30, 2]

        ans = {
            'tag_k': tag_k.cpu().numpy(), # [1, 17, 30, 1]
            'loc_k': ind_k.cpu().numpy(), # [1, 17, 30, 2]
            'val_k': val_k.cpu().numpy() # [1, 17, 30]
        }
        return ans

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        #print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)] > tmp[xx, max(yy-1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy] > tmp[max(0, xx-1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def refine(self, det, tag, keypoints):
        """
        Given initial keypoint predictions, we identify missing joints
        :param det: numpy.ndarray of size (17, 128, 128)
        :param tag: numpy.ndarray of size (17, 128, 128) if not flip
        :param keypoints: numpy.ndarray of size (17, 4) if not flip, last dim is (x, y, det score, tag score)
        :return:
        """
        if len(tag.shape) == 3:
            # tag shape: (17, 128, 128, 1)
            tag = tag[:, :, :, None]

        tags = []
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(np.int32)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        ans = []

        for i in range(keypoints.shape[0]):
            # score of joints i at all position
            tmp = det[i, :, :]
            # distance of all tag values with mean tag of current detected people
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
            tmp2 = tmp - np.round(tt)

            # find maximum position
            y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
            xx = x
            yy = y
            # detection score at maximum position
            val = tmp[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(det.shape[0]):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i, :2]
                    keypoints[i, 2] = ans[i, 2]

        return keypoints

    def parse(self, det, tag, adjust=True, refine=True, anns=None):
        # det.shape: [1, 17, 512, w]
        # tag.shape: [1, 17, 512, w, 1]
        # ans: 1 x (2, 17, 4)
        # self.top_k: {'tag_k':(1, 17, 30, 1), 'loc_k':(1, 17, 30, 2), 'val_k':(1, 17, 30)}
        ans = self.match(**self.top_k(det, tag))

        if anns is not None and ((len(ans)==0 or (len(ans)==1 and len(ans[0])==0)) and len(anns)>0):
            pass
            #print(f'matching failed. gt annot num: {len(anns)}, grouped num: {len(ans)}')
            #import ipdb; ipdb.set_trace()

        if adjust:
            ans = self.adjust(ans, det) # 1 x (n_prediction, 17, 4)

        # scores: [2]
        scores = [i[:, 2].mean() for i in ans[0]]

        if anns is not None:
            if len(scores)==0 and len(anns)>0:
                pass
                #print('adjust failed')
                #import ipdb; ipdb.set_trace()

        if refine:
            ans = ans[0]
            # for every detected person
            for i in range(len(ans)):
                det_numpy = det[0].cpu().numpy() # (17, 512, 768)
                tag_numpy = tag[0].cpu().numpy() # (17, 512, 768, 1)
                if not self.tag_per_joint:
                    tag_numpy = np.tile(
                        tag_numpy, (self.params.num_joints, 1, 1, 1)
                    )
                ans[i] = self.refine(det_numpy, tag_numpy, ans[i]) # [17, 4] wo flip. [17,5] with flip
            ans = [ans] # [n_det, 17, 5]

        return ans, scores
