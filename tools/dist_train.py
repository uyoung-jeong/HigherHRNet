# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))
import hhrnet.tools._init_paths as _init_paths
import hhrnet.lib.models as models

from hhrnet.lib.config import cfg
from hhrnet.lib.config import update_config
from hhrnet.lib.core.loss import MultiLossFactory
from hhrnet.lib.core.trainer import do_train
from hhrnet.lib.dataset import make_dataloader
from hhrnet.lib.fp16_utils.fp16util import network_to_half
from hhrnet.lib.fp16_utils.fp16_optimizer import FP16_Optimizer
from hhrnet.lib.utils.utils import create_logger
from hhrnet.lib.utils.utils import get_optimizer
from hhrnet.lib.utils.utils import save_checkpoint
from hhrnet.lib.utils.utils import setup_logger

from tqdm import tqdm
import torchvision.transforms
from hhrnet.lib.core.group import HeatmapParser
from hhrnet.lib.dataset import make_test_dataloader
from hhrnet.lib.utils.transforms import resize_align_multi_scale
from hhrnet.lib.utils.transforms import get_final_preds
from hhrnet.lib.utils.transforms import get_multi_scale_size
from hhrnet.lib.core.inference import get_multi_stage_outputs
from hhrnet.lib.core.inference import aggregate_results
from hhrnet.lib.utils.vis import save_valid_image

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    logger.info(final_output_dir)
    logger.info(tb_log_dir)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or cfg.MULTIPROCESSING_DISTRIBUTED

    ngpus_per_node = torch.cuda.device_count()
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, final_output_dir, tb_log_dir)
        )
    else:
        # Simply call main_worker function
        main_worker(
            ','.join([str(i) for i in cfg.GPUS]),
            ngpus_per_node,
            args,
            final_output_dir,
            tb_log_dir
        )


def main_worker(
        gpu, ngpus_per_node, args, final_output_dir, tb_log_dir
):
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.FP16.ENABLED:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if cfg.FP16.STATIC_LOSS_SCALE != 1.0:
        if not cfg.FP16.ENABLED:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    update_config(cfg, args)

    # setup logger
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        this_dir = os.path.dirname(__file__)
        shutil.copy2(
            os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
            final_output_dir
        )

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        dump_input = torch.rand(
            (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
        )
        writer_dict['writer'].add_graph(model, (dump_input, ))
        # logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.MODEL.SYNC_BN and not args.distributed:
        print('Warning: Sync BatchNorm is only supported in distributed training.')

    if args.distributed:
        if cfg.MODEL.SYNC_BN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            if isinstance(args.gpu, str):
                args.gpu = int(args.gpu)
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        if isinstance(args.gpu, str):
            args.gpu = int(args.gpu)
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    loss_factory = MultiLossFactory(cfg).cuda()

    # Data loading code
    train_loader = make_dataloader(
        cfg, is_train=True, distributed=args.distributed
    )
    logger.info(train_loader.dataset)

    # validation data
    val_data_loader, val_dataset = make_test_dataloader(cfg)
    parser = HeatmapParser(cfg)

    if cfg.MODEL.NAME == 'pose_hourglass':
        val_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),])
    else:
        val_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    best_perf = -1
    best_model = False
    best_epoch = -1
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)

    if cfg.FP16.ENABLED:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=cfg.FP16.STATIC_LOSS_SCALE,
            dynamic_loss_scale=cfg.FP16.DYNAMIC_LOSS_SCALE
        )

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')
    if not os.path.exists(checkpoint_file) and (cfg.AUTO_RESUME and os.path.exists(cfg.TRAIN.CHECKPOINT)):
        checkpoint_file = cfg.TRAIN.CHECKPOINT

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    if cfg.FP16.ENABLED:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer.optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train one epoch
        do_train(cfg, model, train_loader, loss_factory, optimizer, epoch,
                 final_output_dir, tb_log_dir, writer_dict, fp16=cfg.FP16.ENABLED)
        # In PyTorch 1.1.0 and later, you should call `lr_scheduler.step()` after `optimizer.step()`.
        lr_scheduler.step()

        # validation
        if args.rank == 0 and epoch % 2 == 0:
            all_preds = []
            all_scores = []

            pbar = tqdm(total=len(val_dataset), dynamic_ncols=True)
            # len(annos)==n
            # annos[i]: {'segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 'image_id', 'bbox', 'category_id', 'id'}
            n_failed_group = 0
            for i, (images, annos) in enumerate(val_data_loader):
                assert 1 == images.size(0), 'Test batch size should be 1'

                image = images[0].cpu().numpy()
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
                        image_resized = val_transforms(image_resized)
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

                    if torch.sum(torch.isnan(tags))>0:
                        print('nan value in tags')
                        import ipdb; ipdb.set_trace()

                    grouped, scores = parser.parse(
                        final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE, anns=annos
                    )

                    if annos is not None and (len(scores)==0 and len(annos)>0):
                        n_failed_group += 1

                    final_results = get_final_preds(
                        grouped, center, scale,
                        [final_heatmaps.size(3), final_heatmaps.size(2)]
                    )

                pbar.update()
                if i % cfg.PRINT_FREQ == 0:
                    prefix = '{}_{}'.format(os.path.join(final_output_dir, 'result_valid'), i)
                    # logger.info('=> write {}'.format(prefix))
                    save_valid_image(image, final_results, '{}.jpg'.format(prefix), dataset=val_dataset.name)
                    # save_debug_images(cfg, image_resized, None, None, outputs, prefix)
                if len(scores)>0:
                    all_preds.append(final_results)
                    all_scores.append(scores)
            pbar.close()

            print(f'{n_failed_group}/{len(val_dataset)} failed grouping')

            perf_indicator = -1
            writer = writer_dict['writer']
            if len(all_scores)>0:
                name_values, _ = val_dataset.evaluate(
                    cfg, all_preds, all_scores, final_output_dir
                )

                # best case detection
                perf_indicator = name_values['AP']
                writer.add_scalar('val/AP', name_values['AP'], epoch)
            else:
                writer.add_scalar('val/AP', -1, epoch)

            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_epoch = epoch
                best_model = True
            else:
                best_model = False

            # save checkpoint
            if not cfg.MULTIPROCESSING_DISTRIBUTED or (
                    cfg.MULTIPROCESSING_DISTRIBUTED
                    and args.rank == 0
            ):
                logger.info('=> saving checkpoint to {}'.format(final_output_dir))
                best_state_dict = model.module.state_dict() if cfg.MULTIPROCESSING_DISTRIBUTED else model.state_dict()
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cfg.MODEL.NAME,
                    'state_dict': model.state_dict(),
                    'best_state_dict': best_state_dict,
                    'perf': perf_indicator,
                    'optimizer': optimizer.state_dict(),
                }, best_model, final_output_dir)

    if args.rank == 0:
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state{}.pth.tar'.format(gpu)
        )

        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict['writer'].close()

        print(f'best AP: {best_perf}, best epoch: {best_epoch}')

if __name__ == '__main__':
    main()
