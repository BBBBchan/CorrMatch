import argparse
import logging
import os
import pprint

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from evaluate import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
from util.thresh_helper import ThreshController
from einops import rearrange
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.enabled = True
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    init_seeds(0, False)

    model = DeepLabV3Plus(cfg)
    # sam = sam_model_registry["vit_b"](checkpoint="sam/checkpoints/sam_vit_b.pth")
    # sam.cuda()

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    criterion_kl = nn.KLDivLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=4, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=4, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    thresh_controller = ThreshController(nclass=21, momentum=0.999, thresh_init=cfg['thresh_init'])

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_loss_kl = 0.0
        total_loss_corr_ce, total_loss_corr_u = 0.0, 0.0
        total_mask_ratio = 0.0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        if rank == 0:
            tbar = tqdm(total=len(trainloader_l))

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
                (img_u_w_mix, img_u_s1_mix, _, ignore_mask_mix, _, _)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, ignore_mask = img_u_s1.cuda(), ignore_mask.cuda()
            cutmix_box1 = cutmix_box1.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix = img_u_s1_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            b, c, h, w = img_x.shape

            with torch.no_grad():
                model.eval()
                res_u_w_mix = model(img_u_w_mix, need_fp=False, use_corr=False)
                pred_u_w_mix = res_u_w_mix['out'].detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                    img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            res_w = model(torch.cat((img_x, img_u_w)), need_fp=True, use_corr=True)

            preds = res_w['out']
            preds_fp = res_w['out_fp']
            preds_corr = res_w['corr_out']
            preds_corr_map = res_w['corr_map'].detach()
            pred_x_corr, pred_u_w_corr = preds_corr.split([num_lb, num_ulb])
            pred_u_w_corr_map = preds_corr_map[num_lb:]
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            res_s = model(img_u_s1, need_fp=False, use_corr=True)
            pred_u_s1 = res_s['out']
            pred_u_s1_corr = res_s['corr_out']

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.detach().softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.detach().argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            corr_map_u_w_cutmixed1 = pred_u_w_corr_map.clone()
            b_sample, c_sample, _, _ = corr_map_u_w_cutmixed1.shape

            cutmix_box1_map = (cutmix_box1 == 1)

            mask_u_w_cutmixed1[cutmix_box1_map] = mask_u_w_mix[cutmix_box1_map]
            mask_u_w_cutmixed1_copy = mask_u_w_cutmixed1.clone()
            conf_u_w_cutmixed1[cutmix_box1_map] = conf_u_w_mix[cutmix_box1_map]
            ignore_mask_cutmixed1[cutmix_box1_map] = ignore_mask_mix[cutmix_box1_map]
            cutmix_box1_sample = rearrange(cutmix_box1_map, 'n h w -> n 1 h w')
            ignore_mask_cutmixed1_sample = rearrange((ignore_mask_cutmixed1 != 255), 'n h w -> n 1 h w')
            corr_map_u_w_cutmixed1 = (corr_map_u_w_cutmixed1 * ~cutmix_box1_sample * ignore_mask_cutmixed1_sample).bool()

            thresh_controller.thresh_update(pred_u_w.detach(), ignore_mask_cutmixed1, update_g=True)
            thresh_global = thresh_controller.get_thresh_global()

            conf_fliter_u_w = ((conf_u_w_cutmixed1 >= thresh_global) & (ignore_mask_cutmixed1 != 255))
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w.clone()
            conf_fliter_u_w_sample = rearrange(conf_fliter_u_w_without_cutmix, 'n h w -> n 1 h w')

            segments = (corr_map_u_w_cutmixed1 * conf_fliter_u_w_sample).bool()

            for img_idx in range(b_sample):
                for segment_idx in range(c_sample):

                    segment = segments[img_idx, segment_idx]
                    segment_ori = corr_map_u_w_cutmixed1[img_idx, segment_idx]
                    high_conf_ratio = torch.sum(segment)/torch.sum(segment_ori)
                    if torch.sum(segment) == 0 or high_conf_ratio < thresh_global:
                        continue
                    unique_cls, count = torch.unique(mask_u_w_cutmixed1[img_idx][segment==1], return_counts=True)

                    if torch.max(count) / torch.sum(count) > thresh_global:
                        top_class = unique_cls[torch.argmax(count)]
                        mask_u_w_cutmixed1[img_idx][segment_ori==1] = top_class
                        conf_fliter_u_w_without_cutmix[img_idx] = conf_fliter_u_w_without_cutmix[img_idx] | segment_ori
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w_without_cutmix | conf_fliter_u_w


            loss_x = criterion_l(pred_x, mask_x)
            loss_x_corr = criterion_l(pred_x_corr, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * conf_fliter_u_w_without_cutmix
            loss_u_s1 = torch.sum(loss_u_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()

            loss_u_corr_s1 = criterion_u(pred_u_s1_corr, mask_u_w_cutmixed1)
            loss_u_corr_s1 = loss_u_corr_s1 * conf_fliter_u_w_without_cutmix
            loss_u_corr_s1 = torch.sum(loss_u_corr_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_corr_s = loss_u_corr_s1

            loss_u_corr_w = criterion_u(pred_u_w_corr, mask_u_w)
            loss_u_corr_w = loss_u_corr_w * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_corr_w = torch.sum(loss_u_corr_w) / torch.sum(ignore_mask != 255).item()
            loss_u_corr = 0.5 * (loss_u_corr_s + loss_u_corr_w)

            softmax_pred_u_w = F.softmax(pred_u_w.detach(), dim=1)
            logsoftmax_pred_u_s1 = F.log_softmax(pred_u_s1, dim=1)

            loss_u_kl_sa2wa = criterion_kl(logsoftmax_pred_u_s1, softmax_pred_u_w)
            loss_u_kl_sa2wa = torch.sum(loss_u_kl_sa2wa, dim=1) * conf_fliter_u_w
            loss_u_kl_sa2wa = torch.sum(loss_u_kl_sa2wa) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_kl = loss_u_kl_sa2wa

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_w_fp = torch.sum(loss_u_w_fp) / torch.sum(ignore_mask != 255).item()

            loss = ( 0.5 * loss_x + 0.5 * loss_x_corr + loss_u_s1 * 0.25 + loss_u_kl * 0.25 + loss_u_w_fp * 0.25 + 0.25 * loss_u_corr) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_s += loss_u_s1.item()
            total_loss_kl += loss_u_kl.item()
            total_loss_w_fp += loss_u_w_fp.item()
            total_loss_corr_ce += loss_x_corr.item()
            total_loss_corr_u += loss_u_corr.item()
            total_mask_ratio += ((conf_u_w >= thresh_global) & (ignore_mask != 255)).sum().item() / \
                                (ignore_mask != 255).sum().item()

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                tbar.set_description(' Total loss: {:.3f}, Loss x: {:.3f}, loss_corr_ce: {:.3f} '
                                     'Loss s: {:.3f}, Loss w_fp: {:.3f},  Mask: {:.3f}, loss_corr_u: {:.3f}'.format(
                    total_loss / (i + 1), total_loss_x / (i + 1), total_loss_corr_ce / (i + 1), total_loss_s / (i + 1),
                    total_loss_w_fp / (i + 1), total_mask_ratio / (i + 1), total_loss_corr_u / (i + 1)))
                tbar.update(1)

        if rank == 0:
            tbar.close()

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        torch.cuda.empty_cache()
        res_val = evaluate(model, valloader, eval_mode, cfg)
        mIOU = res_val['mIOU']
        class_IOU = res_val['iou_class']
        torch.distributed.barrier()

        if rank == 0:
            logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.4f} \n'.format(eval_mode, mIOU))
            logger.info('***** ClassIOU ***** >>>> \n{}\n'.format(class_IOU))

        if mIOU > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(), os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], mIOU)))
        torch.distributed.barrier()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
