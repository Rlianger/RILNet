import logging
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from monai.losses import FocalLoss, DiceLoss
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


class BodyPartAttentionLoss(nn.Module):
    """ A body part attention loss as described in our paper
    'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
    Source: https://github.com/VlSomers/bpbreid
    """

    def __init__(self, loss_type='cl', label_smoothing=0.1, use_gpu=False):
        super().__init__()
        self.pred_accuracy = Accuracy(top_k=1)
        if use_gpu:
            self.pred_accuracy = self.pred_accuracy.cuda()
        if loss_type == 'cl':
            self.part_prediction_loss_1 = CrossEntropyLoss(label_smoothing=label_smoothing)
        elif loss_type == 'fl':
            self.part_prediction_loss = FocalLoss(to_onehot_y=True, gamma=1.0)
        elif loss_type == 'dl':
            self.part_prediction_loss = DiceLoss(to_onehot_y=True, softmax=True)
        else:
            raise ValueError("Loss {} for part prediction is not supported".format(loss_type))

    def forward(self, pixels_cls_scores, targets):
        """ Compute loss for body part attention prediction.
            Args:
                pixels_cls_scores [N, K, H, W]
                targets [N, H, W]
            Returns:
        """
        loss_summary = {}
        loss_summary['pixls'] = OrderedDict()
        pixels_cls_loss, pixels_cls_accuracy = self.compute_pixels_cls_loss(pixels_cls_scores, targets)
        loss_summary['pixls']['c'] = pixels_cls_loss
        loss_summary['pixls']['a'] = pixels_cls_accuracy
        return pixels_cls_loss, loss_summary

    def compute_pixels_cls_loss(self, pixels_cls_scores, targets):
        if pixels_cls_scores.is_cuda:
            targets = targets.cuda()
        pixels_cls_score_targets = targets.flatten()  # [N*Hf*Wf]
        pixels_cls_scores = pixels_cls_scores.permute(0, 2, 3, 1).flatten(0, 2)  # [N*Hf*Wf, M]
        loss = self.part_prediction_loss_1(pixels_cls_scores, pixels_cls_score_targets)
        accuracy = self.pred_accuracy(pixels_cls_scores, pixels_cls_score_targets)
        return loss, accuracy.item()


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             Moptimizer,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    name = cfg.DATASETS.NAMES[1]  # occluded reid 1 | market 0
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    bpaLoss = BodyPartAttentionLoss('cl', use_gpu=True)
    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    logger.info('datasets name is ' + name)
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    maxCmc = 0
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, data in enumerate(train_loader):
            img, target_masks, target, imgs_path, target_cam, target_view = get_train_data(data, cfg.MODEL.MASK_NUM)
            img = img.to(device)
            target = target.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            ############ Feature extractor ###########################
            optimizer.zero_grad()
            with (amp.autocast(enabled=True)):
                score, Mscore, feat, orth_proto, partial, Pscore, body_cls_scores = model(img, target,
                                                                                          cam_label=target_cam,
                                                                                          view_label=target_view)

                body_cls_score_targets = target_masks.argmax(dim=1)
                bpbloss, loss_summary = bpaLoss(body_cls_scores,
                                                body_cls_score_targets)  # [b. 10 , 768, 1]  -> [8,64,32]    || [b,9, 96, 32] , [b, 96,32]
                loss = loss_fn(score, Mscore, feat, orth_proto, epoch, target, target_cam)
                p_loss = loss_fn(Pscore, Mscore, partial, orth_proto, epoch, target, partial=True)
                lamda = 0.5 if name == "occluded_duke" else 0.01 # lamda
                lamda2 = 0.5 if name == "occluded_duke" else 0.01 # omiga
                # lamda = 0.01
                # lamda2 = 0.01
                loss += lamda * p_loss
                loss +=  bpbloss * lamda2

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            ############ Mask Generator ###############################
            Moptimizer.zero_grad()
            with (amp.autocast(enabled=True)):
                score, Mscore, feat, orth_proto, partial, Pscore, body_cls_scores = model(img, target,
                                                                                          cam_label=target_cam,
                                                                                          view_label=target_view)
                loss = loss_fn(score, Mscore, feat, orth_proto, epoch, target, target_cam)
                bpbloss, loss_summary = bpaLoss(body_cls_scores,
                                                body_cls_score_targets)
                p_loss = loss_fn(Pscore, Mscore, partial, orth_proto, epoch, target, partial=True)
                loss += lamda * p_loss
                loss += lamda2 * bpbloss

            scaler.scale(loss).backward()

            scaler.step(Moptimizer)
            scaler.update()
            ###########################################################
            '''if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()'''
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                gallery = val_loader[name]['gallery']
                query = val_loader[name]['query']
                for n_iter, data in enumerate(query):
                    img, vid, camid, camids, target_view, path = get_test_data(data)
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, mask = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, mask, vid, camid,path))
                for n_iter, data in enumerate(gallery):
                    img, vid, camid, camids, target_view, path = get_test_data(data)
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, mask = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, mask, vid, camid, path))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
                # save best
                if cmc[0] > maxCmc:
                    maxCmc = cmc[0]
                    torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'BestModel.pth'))
                    logger.info("save best model - Epoch:{}".format(epoch))



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    name = cfg.DATASETS.NAMES[1]
    gallery = val_loader[name]['gallery']
    query = val_loader[name]['query']
    for n_iter, data in enumerate(query):
        img, vid, camid, camids, target_view, path = get_test_data(data)
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat, mask = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, mask, vid, camid,path))
            img_path_list.extend(path)
    for n_iter, data in enumerate(gallery):
        img, vid, camid, camids, target_view, path = get_test_data(data)
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat, mask = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, mask, vid, camid,path))
            img_path_list.extend(path)
    # for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
    #     with torch.no_grad():
    #         img = img.to(device)
    #         camids = camids.to(device)
    #         target_view = target_view.to(device)
    #         feat, mask = model(img, cam_label=camids, view_label=target_view)
    #         evaluator.update((feat, mask, pid, camid))
    #         img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def get_train_data(data, masks_part=0):
    imgs = data['image']
    imgs_path = data['img_path']
    masks = data['mask'] if 'mask' in data else None
    pids = data['pid']
    target_cams = data['camid']
    view_cams = data['viewid']
    if masks is not None:
        assert masks.shape[1] == (masks_part + 1)

    return imgs, masks, pids, imgs_path, target_cams, view_cams


def get_test_data(data):
    imgs = data['image']
    pids = data['pid']
    camid = data['camid']
    camids = torch.tensor(camid, dtype=torch.int64)
    viewid = data['viewid']
    imgs_path = data['img_path']
    return imgs, pids, camid, camids, viewid, imgs_path