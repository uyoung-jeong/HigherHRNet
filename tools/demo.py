# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" single scale
python tools/demo.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    --input demo_images \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth \
    TEST.FLIP_TEST False
"""
""" multi-scale
python tools/demo.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    --input demo_images \
    --score_thres 0.15 \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth \
    TEST.SCALE_FACTOR '[0.5, 1.0, 2.0]' \
    TEST.FLIP_TEST False
"""

import argparse
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

import hhrnet.tools._init_paths as _init_paths
import hhrnet.lib.models as models

from hhrnet.lib.config import cfg
from hhrnet.lib.config import check_config
from hhrnet.lib.config import update_config
from hhrnet.lib.core.inference import get_multi_stage_outputs
from hhrnet.lib.core.inference import aggregate_results
from hhrnet.lib.core.group import HeatmapParser
from hhrnet.lib.dataset import make_test_dataloader
from hhrnet.lib.fp16_utils.fp16util import network_to_half
from hhrnet.lib.utils.utils import create_logger
from hhrnet.lib.utils.utils import get_model_summary
from hhrnet.lib.utils.vis import save_debug_images
from hhrnet.lib.utils.vis import save_valid_image
from hhrnet.lib.utils.transforms import resize_align_multi_scale
from hhrnet.lib.utils.transforms import get_final_preds
from hhrnet.lib.utils.transforms import get_multi_scale_size

from hhrnet.lib.dataset.COCODataset import CocoDataset as coco
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import cv2

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--input', help='input directory or file', type=str)

    parser.add_argument('--score_thres', help='score threshold', type=float, default=0.2)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args

def evaluate(self, cfg, preds, scores, output_dir, file_names,
             *args, **kwargs):
    '''
    Perform evaluation on COCO keypoint task
    :param cfg: cfg dictionary
    :param preds: prediction
    :param output_dir: output directory
    :param args:
    :param kwargs:
    :return:
    '''
    res_folder = os.path.join(output_dir, 'results')
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    res_file = os.path.join(
        res_folder, 'keypoints_demo_results.json')

    # preds is a list of: image x person x (keypoints)
    # keypoints: num_joints * 4 (x, y, score, tag)
    kpts = defaultdict(list)
    for idx, _kpts in enumerate(preds):
        file_name = file_names[idx]
        for idx_kpt, kpt in enumerate(_kpts):
            area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
            kpt = self.processKeypoints(kpt)
            if cfg.DATASET.WITH_CENTER and not cfg.TEST.IGNORE_CENTER:
                kpt = kpt[:-1]

            kpts[file_name].append(
                {
                    'keypoints': kpt[:, 0:3],
                    'score': scores[idx][idx_kpt],
                    'tags': kpt[:, 3],
                    'image': file_name,
                    'area': area
                }
            )

    # rescoring and oks nms
    oks_nmsed_kpts = []
    # image x person x (keypoints)
    for img in kpts.keys():
        # person x (keypoints)
        img_kpts = kpts[img]
        # person x (keypoints)
        # do not use nms, keep all detections
        keep = []
        if len(keep) == 0:
            oks_nmsed_kpts.append(img_kpts)
        else:
            oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

    self._write_coco_keypoint_results(
        oks_nmsed_kpts, res_file
    )

    return oks_nmsed_kpts

def main(args=None):
    if args is None:
        args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    print(f'hhrnet output dir: {final_output_dir}')

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    dump_input = torch.rand(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    )
    #logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()


    #data_loader, test_dataset = make_test_dataloader(cfg)

    dummy_dataset = eval(cfg.DATASET.DATASET_TEST)(
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST,
        cfg.DATASET.DATA_FORMAT,
        None
    )

    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    # load images
    input_path = args.input
    files = []
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        res_files = [e for e in os.listdir(input_path) if os.path.isfile(os.path.join(input_path,e))]
        ext_filters = ['jpg', 'jpeg', 'png', 'PNG']
        img_files = [e for e in res_files if e.split('.')[-1] in ext_filters]
        files = [os.path.join(input_path,e) for e in img_files]
    else:
        print(f'{input_path} seems to be an inappropriate path')
    files.sort()
    
    parser = HeatmapParser(cfg)
    all_preds = []
    all_scores = []
    for i, fname in enumerate(tqdm(files)):
        # load image
        image = cv2.imread(fname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )

            # filter results by score threshold
            if len(final_results)==0:
                continue
            final_results = np.stack(final_results)
            scores = np.array(scores)
            valid_indices = scores>args.score_thres
            final_results = final_results[valid_indices]
            scores = scores[valid_indices]

            prefix = '{}_{}'.format(os.path.join(final_output_dir, 'result_demo'), fname.split(os.sep)[-1])
            save_valid_image(image, final_results, prefix, dataset=dummy_dataset.name)

        all_preds.append(final_results) # n_result x 17 x 5
        all_scores.append(scores)

    oks_nmsed_kpts = evaluate(
        dummy_dataset, cfg, all_preds, all_scores, final_output_dir, files
    )

    return oks_nmsed_kpts


if __name__ == '__main__':
    main()
