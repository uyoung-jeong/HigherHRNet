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
import os
import time

from tqdm import tqdm

import torch.cuda.amp as amp

from hhrnet.lib.utils.utils import AverageMeter
from hhrnet.lib.utils.vis import save_debug_images


def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict, fp16=False, scaler=None):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmaps_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    push_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    pull_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    prior_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]

    # switch to train mode
    model.train()

    end = time.time()
    pbar = tqdm(total=len(data_loader), dynamic_ncols=True) if cfg.RANK == 0 else None
    for i, (images, heatmaps, masks, joints) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # visualize joint
        #if not cfg.MULTIPROCESSING_DISTRIBUTED:
            # from utils.vis import visualize_img_joint_train
            #visualize_img_joint_train(cfg, images, joints)

        if images.get_device()<0: # cpu
            images = images.cuda(non_blocking=True)

        optimizer.zero_grad()
        # compute output
        outputs = model(images) # [[b, 34, 128, 128], [b,17,256,256]]


        heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps)) # [[b, 17, 128, 128], [b, 12, 17, 256, 256]]
        masks = list(map(lambda x: x.cuda(non_blocking=True), masks)) # [[b, 128, 128], [b, 256, 256]]
        joints = list(map(lambda x: x.cuda(non_blocking=True), joints)) # [[b,MAX_NUM_PEOPLE,17,2], [b,MAX_NUM_PEOPLE,17,2]]

        # loss = loss_factory(outputs, heatmaps, masks)
        heatmaps_losses, push_losses, pull_losses, prior_losses = \
            loss_factory(outputs, heatmaps, masks, joints)

        loss = 0
        for idx in range(cfg.LOSS.NUM_STAGES):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                heatmaps_loss_meter[idx].update(
                    heatmaps_loss.item(), images.size(0)
                )
                loss = loss + heatmaps_loss
                if push_losses[idx] is not None:
                    push_loss = push_losses[idx].mean(dim=0)
                    push_loss_meter[idx].update(
                        push_loss.item(), images.size(0)
                    )
                    loss = loss + push_loss
                if pull_losses[idx] is not None:
                    pull_loss = pull_losses[idx].mean(dim=0)
                    pull_loss_meter[idx].update(
                        pull_loss.item(), images.size(0)
                    )
                    loss = loss + pull_loss
                if prior_losses[idx] is not None:
                    prior_loss = prior_losses[idx].mean(dim=0)
                    prior_loss_meter[idx].update(
                        prior_loss.item(), images.size(0)
                    )
                    loss = loss + prior_loss

        # compute gradient and do update step
        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if pbar is not None:
            pbar.update(1)

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            """
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.2f}s)\t' \
                  '{heatmaps_loss}{push_loss}{pull_loss}{prior_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(heatmaps_loss_meter, 'heatmaps'),
                      push_loss=_get_loss_info(push_loss_meter, 'push'),
                      pull_loss=_get_loss_info(pull_loss_meter, 'pull'),
                      prior_loss=_get_loss_info(prior_loss_meter, 'prior')
                  )
            """
            msg = f"[{epoch}][{i}/{len(data_loader)}], {_get_loss_info(heatmaps_loss_meter, 'heatmaps')}"
            msg += f", {_get_loss_info(push_loss_meter, 'push')},{_get_loss_info(pull_loss_meter, 'pull')}"
            #msg += f", {_get_loss_info(prior_loss_meter, 'prior')}"
            #logger.info(msg)
            if pbar is not None:
                pbar.set_description(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            for idx in range(cfg.LOSS.NUM_STAGES):
                writer.add_scalar(
                    'train_stage{}_heatmaps_loss'.format(idx),
                    heatmaps_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_push_loss'.format(idx),
                    push_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_pull_loss'.format(idx),
                    pull_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_prior_loss'.format(idx),
                    prior_loss_meter[idx].val,
                    global_steps
                )
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            for scale_idx in range(len(outputs)):
                prefix_scale = prefix + '_output_{}'.format(
                    cfg.DATASET.OUTPUT_SIZE[scale_idx]
                )
                save_debug_images(
                    cfg, images, heatmaps[scale_idx], masks[scale_idx],
                    outputs[scale_idx], prefix_scale
                )
    if pbar is not None:
        pbar.close()

def do_train_amp(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict, fp16=False, scaler=None):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmaps_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    push_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    pull_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    prior_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]

    # switch to train mode
    model.train()

    end = time.time()
    pbar = tqdm(total=len(data_loader), dynamic_ncols=True) if cfg.RANK == 0 else None
    for i, (images, heatmaps, masks, joints) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # visualize joint
        #if not cfg.MULTIPROCESSING_DISTRIBUTED:
            # from utils.vis import visualize_img_joint_train
            #visualize_img_joint_train(cfg, images, joints)

        if images.get_device()<0: # cpu
            images = images.cuda(non_blocking=True)

        optimizer.zero_grad()
        # compute output
        with amp.autocast(cfg.FP16.ENABLED):
            outputs = model(images) # [[b, 34, 128, 128], [b,17,256,256]]


            heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps)) # [[b, 17, 128, 128], [b, 12, 17, 256, 256]]
            masks = list(map(lambda x: x.cuda(non_blocking=True), masks)) # [[b, 128, 128], [b, 256, 256]]
            joints = list(map(lambda x: x.cuda(non_blocking=True), joints)) # [[b,MAX_NUM_PEOPLE,17,2], [b,MAX_NUM_PEOPLE,17,2]]

            # loss = loss_factory(outputs, heatmaps, masks)
            heatmaps_losses, push_losses, pull_losses, prior_losses = \
                loss_factory(outputs, heatmaps, masks, joints)

            loss = 0
            for idx in range(cfg.LOSS.NUM_STAGES):
                if heatmaps_losses[idx] is not None:
                    heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                    heatmaps_loss_meter[idx].update(
                        heatmaps_loss.item(), images.size(0)
                    )
                    loss = loss + heatmaps_loss
                    if push_losses[idx] is not None:
                        push_loss = push_losses[idx].mean(dim=0)
                        push_loss_meter[idx].update(
                            push_loss.item(), images.size(0)
                        )
                        loss = loss + push_loss
                    if pull_losses[idx] is not None:
                        pull_loss = pull_losses[idx].mean(dim=0)
                        pull_loss_meter[idx].update(
                            pull_loss.item(), images.size(0)
                        )
                        loss = loss + pull_loss
                    if prior_losses[idx] is not None:
                        prior_loss = prior_losses[idx].mean(dim=0)
                        prior_loss_meter[idx].update(
                            prior_loss.item(), images.size(0)
                        )
                        loss = loss + prior_loss

        # backward
        with amp.autocast(False):
            loss = scaler.scale(loss)
            loss.backward()
            scaler.step(optimizer)
        scaler.update(1024.)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if pbar is not None:
            pbar.update(1)

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            """
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.2f}s)\t' \
                  '{heatmaps_loss}{push_loss}{pull_loss}{prior_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(heatmaps_loss_meter, 'heatmaps'),
                      push_loss=_get_loss_info(push_loss_meter, 'push'),
                      pull_loss=_get_loss_info(pull_loss_meter, 'pull'),
                      prior_loss=_get_loss_info(prior_loss_meter, 'prior')
                  )
            """
            msg = f"[{epoch}][{i}/{len(data_loader)}], {_get_loss_info(heatmaps_loss_meter, 'heatmaps')}"
            msg += f", {_get_loss_info(push_loss_meter, 'push')},{_get_loss_info(pull_loss_meter, 'pull')}"
            #msg += f", {_get_loss_info(prior_loss_meter, 'prior')}"
            #logger.info(msg)
            if pbar is not None:
                pbar.set_description(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            for idx in range(cfg.LOSS.NUM_STAGES):
                writer.add_scalar(
                    'train_stage{}_heatmaps_loss'.format(idx),
                    heatmaps_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_push_loss'.format(idx),
                    push_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_pull_loss'.format(idx),
                    pull_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_prior_loss'.format(idx),
                    prior_loss_meter[idx].val,
                    global_steps
                )
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            for scale_idx in range(len(outputs)):
                prefix_scale = prefix + '_output_{}'.format(
                    cfg.DATASET.OUTPUT_SIZE[scale_idx]
                )
                save_debug_images(
                    cfg, images, heatmaps[scale_idx], masks[scale_idx],
                    outputs[scale_idx], prefix_scale
                )
    if pbar is not None:
        pbar.close()

def _get_loss_info(loss_meters, loss_name):
    msg = ''
    for i, meter in enumerate(loss_meters):
        msg += 'Stage{i}-{name}: {meter.val:.3e} ({meter.avg:.3e})'.format(
            i=i, name=loss_name, meter=meter
        )

    return msg
